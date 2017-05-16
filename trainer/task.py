import argparse
import json
import os
import tensorflow as tf
import threading
import model

tf.logging.set_verbosity(tf.logging.INFO)


class EvaluationRunHook(tf.train.SessionRunHook):
    """
    Hook responsible to running an evaluation on the validation set during training.
    """

    def __init__(self,
                 checkpoint_dir,
                 graph,
                 eval_frequency,
                 **kwargs):
        """
        
        :param checkpoint_dir: dir where the checkpoints are saved
        :param graph: evaluation graph
        :param eval_frequency: runs evaluation every n checkpoints saved
        :param kwargs: 
        """
        self._checkpoint_dir = checkpoint_dir
        self._kwargs = kwargs
        self._eval_every = eval_frequency
        self._latest_checkpoint = None
        self._checkpoints_since_eval = 0
        self._graph = graph
        with graph.as_default():
            self._saver = tf.train.Saver()
            self._gs = tf.contrib.framework.get_or_create_global_step()

        self._eval_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()
        self._file_writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval'), graph=graph)

    def after_run(self, run_context, run_values):
        self._update_latest_checkpoint()

        if self._eval_lock.acquire(False):
            try:
                if self._checkpoints_since_eval >= self._eval_every:
                    self._checkpoints_since_eval = 0
                    self._run_eval()
            finally:
                self._eval_lock.release()

    def _update_latest_checkpoint(self):
        """Update the latest checkpoint file created in the output dir."""
        if self._checkpoint_lock.acquire(False):
            try:
                latest = tf.train.latest_checkpoint(self._checkpoint_dir)
                if not latest == self._latest_checkpoint:
                    self._checkpoints_since_eval += 1
                    self._latest_checkpoint = latest
            finally:
                self._checkpoint_lock.release()

    def end(self, session):
        """Called at then end of session to make sure we always evaluate."""
        self._update_latest_checkpoint()
        with self._eval_lock:
            self._run_eval()

    def _run_eval(self):
        """
        Run model evaluation and generate summaries.
        """
        coord = tf.train.Coordinator(clean_stop_exception_types=(
            tf.errors.CancelledError, tf.errors.OutOfRangeError))

        with tf.Session(graph=self._graph) as session:
            # Restores previously saved variables from latest checkpoint
            self._saver.restore(session, self._latest_checkpoint)
            session.run([
                tf.tables_initializer(),
                tf.local_variables_initializer(),
            ])
            threads = tf.train.start_queue_runners(coord=coord, sess=session)
            train_step = session.run(self._gs)
            tf.logging.info('Starting Evaluation For Step: {}'.format(train_step))
            op = tf.summary.merge_all()
            sum_res = session.run(op)
            self._file_writer.add_summary(sum_res, global_step=train_step)
            self._file_writer.flush()
            coord.request_stop()
            coord.join(threads)

def build_and_run_exports(latest, job_dir, name, convs, kernels):
    """Given the latest checkpoint file export the saved model.

  Args:
    latest (string): Latest checkpoint file
    job_dir (string): Location of checkpoints and model files
    name (string): Name of the checkpoint to be exported. Used in building the
      export path.
    hidden_units (list): Number of hidden units
    learning_rate (float): Learning rate for the SGD
  """

    prediction_graph = tf.Graph()
    exporter = tf.saved_model.builder.SavedModelBuilder(
        os.path.join(job_dir, 'export', name))
    with prediction_graph.as_default():
        imgs = tf.placeholder(tf.float32, shape=[None, 96, 96, 1])
        prediction_dict = model.model_fn(
            model.PREDICT,
            imgs,
            None,
            convs=convs,
            kernels=kernels,
            learning_rate=None,
        )
        saver = tf.train.Saver()

        inputs_info = {
            'images': tf.saved_model.utils.build_tensor_info(imgs)
        }
        output_info = {
            prediction_dict.keys()[0]: tf.saved_model.utils.build_tensor_info(prediction_dict.values()[0])
        }
        signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs_info,
            outputs=output_info,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

    with tf.Session(graph=prediction_graph) as session:
        session.run([tf.local_variables_initializer(), tf.tables_initializer()])
        saver.restore(session, latest)
        exporter.add_meta_graph_and_variables(
            session,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
            },
        )

    exporter.save()


def run(target,
        is_chief,
        job_dir,
        train_files,
        eval_files,
        batch_size,
        learning_rate,
        num_epochs,
        eval_frequency,
        ckpts_save_freq_sec,
        p_keep,
        convs,
        kernels):
    """
    Runs the training of the model
    
    :param target: Tensorflow server target
    :param is_chief: flag to specify a chief server (True) 
    :param job_dir: output dir for checkpoints and summaries
    :param train_files: .tfrecords file for training
    :param eval_files: .tfrecords file for evaluating (validation set)
    :param batch_size: size of the batch in training and validation
    :param learning_rate: learning rate in SGD
    :param num_epochs: number of epochs in the training data
    :param eval_frequency: evaluation is run every n checkpoints save
    :param ckpts_save_freq_sec: checkpoints are saved every n seconds
    :param p_keep: probability to keep a unit when applying drop out
    :param convs: (string)architecture of DNN, e.g. '32 64 128'
    :param kernels: (list) kernels size for each conv layer. Must be of same length as convs
    """
    convs = [int(lyr) for lyr in convs.split()]
    kernels = [int(k) for k in kernels.split()]
    if len(convs) != len(kernels):
        raise ValueError('A kernel must be provided for each convolutional layer.')
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        imgs, labels = model.input_fn(eval_files, batch_size=batch_size)
        model.model_fn(model.EVAL, imgs, labels, None, convs, kernels, 1.0)

    hook = EvaluationRunHook(job_dir, eval_graph, eval_frequency)

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter()):
            imgs, labels = model.input_fn(train_files,
                                          batch_size=batch_size,
                                          num_epochs=num_epochs)
            g_step, train_op = model.model_fn(
                model.TRAIN,
                imgs,
                labels,
                learning_rate=learning_rate,
                convs=convs,
                kernels=kernels,
                p_keep=p_keep
            )

        hooks = [hook]
        config = tf.ConfigProto()
        config.log_device_placement = False

        with tf.train.MonitoredTrainingSession(
                master=target,
                is_chief=is_chief,
                checkpoint_dir=job_dir,
                config=config,
                hooks=hooks,
                save_checkpoint_secs=ckpts_save_freq_sec,
                save_summaries_steps=50) as session:
            writer = tf.summary.FileWriter(job_dir)
            step = session.run(g_step)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            while True:
                if step % 100 == 0:
                    step, _ = session.run([g_step, train_op],
                                          options=run_options,
                                          run_metadata=run_metadata)
                    writer.add_run_metadata(run_metadata, 'step%d' % step)
                else:
                    step, _ = session.run([g_step, train_op])


def dispatch(*args, **kwargs):
    """Parse TF_CONFIG to cluster_spec and call run() method
  TF_CONFIG environment variable is available when running using
  gcloud either locally or on cloud. It has all the information required
  to create a ClusterSpec which is important for running distributed code.
  """

    tf_config = os.environ.get('TF_CONFIG')
    # If TF_CONFIG is not available run local
    if not tf_config:
        return run('', True, *args, **kwargs)

    tf_config_json = json.loads(tf_config)
    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    if job_name is None or task_index is None:
        return run('', True, *args, **kwargs)

    cluster_spec = tf.train.ClusterSpec(cluster)
    server = tf.train.Server(cluster_spec,
                             job_name=job_name,
                             task_index=task_index)
    if job_name == 'ps':
        server.join()
        return
    elif job_name in ['master', 'worker']:
        return run(server.target, job_name == 'master', *args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-files',
                        required=True,
                        type=str,
                        help='Training files local or GCS', nargs='+')
    parser.add_argument('--eval-files',
                        required=True,
                        type=str,
                        help='Evaluation files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parser.add_argument('--batch-size',
                        type=int,
                        default=40,
                        help='Batch size')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=1,
                        help='Number of epochs')
    parser.add_argument('--eval-frequency',
                        type=int,
                        default=1,
                        help='Evaluates every n checkpoints save')
    parser.add_argument('--ckpts-save-freq-sec',
                        type=int,
                        default=120,
                        help='Save checkpoints every n seconds')
    parser.add_argument('--p-keep',
                        type=float,
                        default=0.5,
                        help='Probability of keeping unit when using dropout')
    parser.add_argument('--convs',
                        type=str,
                        default='32 64 128')
    parser.add_argument('--kernels',
                        type=str,
                        default='3 3 3')

    parse_args, unknown = parser.parse_known_args()

    tf.logging.warn('Unknown arguments: {}'.format(unknown))
    dispatch(**parse_args.__dict__)
