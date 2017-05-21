import argparse
import json
import os

import tensorflow as tf

import model
from trainer.early_stopper import SimpleEarlyStopper
from trainer.evaluation_hook import EvaluationRunHook

tf.logging.set_verbosity(tf.logging.INFO)


def build_and_run_exports(job_dir, train_step, name, convs, kernels):
    """
    Serializes the model to be used for predictions. Basically removes all unnecessary nodes and outputs the graph 
    in a ProfoBuf (.pb) file.
    :param job_dir: dir containing the ckpt files
    :param train_step: the train_step corresponding to the ckpoint we want to build the model from
    :param name: the name of the output file
    :param convs: (list) architecture of the CNN, e.g. [32, 64, 128]
    :param kernels: (list) kernels size for each conv layer. Must be of same length as convs
    """
    ckpt = ckpt_for_step(job_dir, train_step)
    prediction_graph = tf.Graph()
    output_graph = os.path.join(job_dir, 'export', '{}.pb'.format(name))
    output_node_names = "out/fully_connected/BiasAdd"
    with prediction_graph.as_default():
        imgs = tf.placeholder(tf.float32, shape=[None, 96, 96, 1], name='input')
        model.model_fn(
            model.PREDICT,
            imgs,
            None,
            convs=convs,
            kernels=kernels,
            learning_rate=None,
        )
        saver = tf.train.Saver()
    with tf.Session(graph=prediction_graph) as sess:
        saver.restore(sess, ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            prediction_graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())


def ckpt_for_step(job_dir, step):
    """
    The full path to the checkpoint corresponding to a given training step and a given model.
    :param job_dir: The dir of the model.
    :param step: A global training step.
    :return: Full path of the checkpoint.
    """
    state = tf.train.get_checkpoint_state(job_dir)
    for ckpt in state.all_model_checkpoint_paths:
        if '-{}'.format(step) in ckpt:
            return ckpt
    return None


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
        kernels,
        seed):
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
    convs = [int(lyr) for lyr in convs.split('_')]
    kernels = [int(k) for k in kernels.split('_')]
    if len(convs) != len(kernels):
        raise ValueError('A kernel must be provided for each convolutional layer.')
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        tf.set_random_seed(seed)
        imgs, labels = model.input_fn(
            eval_files,
            batch_size=batch_size,
            shuffle=True)
        metric = model.model_fn(model.EVAL, imgs, labels, None, convs, kernels, 1.0)

    early_stopper = SimpleEarlyStopper(chances=5)
    hook = EvaluationRunHook(job_dir,
                             eval_graph,
                             eval_frequency,
                             eval_steps=550 / batch_size,
                             metric=metric,
                             early_stopper=early_stopper)

    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter()):
            imgs, labels = model.input_fn(train_files,
                                          batch_size=batch_size,
                                          num_epochs=num_epochs,
                                          shuffle=True)
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
            while not session.should_stop():
                if step % 100 == 0:
                    step, _ = session.run([g_step, train_op],
                                          options=run_options,
                                          run_metadata=run_metadata)
                    writer.add_run_metadata(run_metadata, 'step%d' % step)
                else:
                    step, _ = session.run([g_step, train_op])
        best_step = early_stopper.best[0]
        build_and_run_exports(job_dir, best_step, 'frozen_model', convs, kernels)


def dispatch(*args, **kwargs):
    """
    Parse TF_CONFIG to cluster_spec and call run() method
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
    parser.add_argument('--seed',
                        type=int,
                        default=67)

    parse_args, unknown = parser.parse_known_args()

    tf.logging.warn('Unknown arguments: {}'.format(unknown))
    dispatch(**parse_args.__dict__)
