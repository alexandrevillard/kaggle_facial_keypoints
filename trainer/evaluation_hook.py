import os
import threading
import numpy as np
import tensorflow as tf


class EvaluationRunHook(tf.train.SessionRunHook):
    """
    Hook responsible to running an evaluation on the validation set during training.
    """

    def __init__(self,
                 checkpoint_dir,
                 graph,
                 eval_frequency,
                 metric,
                 eval_steps=1,
                 early_stopper=None,
                 **kwargs):
        """
        :param checkpoint_dir: dir where the checkpoints are saved
        :param graph: evaluation graph
        :param eval_frequency: runs evaluation every n checkpoints saved
        :param kwargs: 
        """
        self._eval_steps = eval_steps
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
        self._metric = metric
        self._early_stopper = early_stopper

    def after_run(self, run_context, run_values):
        self._update_latest_checkpoint()
        if self._eval_lock.acquire(False):
            try:
                if self._checkpoints_since_eval >= self._eval_every:
                    self._checkpoints_since_eval = 0
                    self._run_eval()
                    if self._should_stop():
                        run_context.request_stop()
                        tf.logging.info('Early stopping has been called')
            finally:
                self._eval_lock.release()

    def _update_latest_checkpoint(self):
        """
        Update the latest checkpoint file created in the output dir.
        """
        if self._checkpoint_lock.acquire(False):
            try:
                latest = tf.train.latest_checkpoint(self._checkpoint_dir)
                if not latest == self._latest_checkpoint:
                    self._checkpoints_since_eval += 1
                    self._latest_checkpoint = latest
            finally:
                self._checkpoint_lock.release()

    def end(self, session):
        """
        Called at then end of session to make sure we always evaluate, unless there's been an early stopping.
        """
        if self._early_stopper is not None and self._early_stopper.should_stop():
            return
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
            self._saver.restore(session, self._latest_checkpoint)
            session.run([
                tf.tables_initializer(),
                tf.local_variables_initializer(),
            ])
            threads = tf.train.start_queue_runners(coord=coord, sess=session)
            train_step = session.run(self._gs)
            tf.logging.info('Starting Evaluation For Step: {}'.format(train_step))
            losses = np.zeros(self._eval_steps, np.float32)
            with coord.stop_on_exception():
                eval_step = 0
                while eval_step < self._eval_steps:
                    [metric_res] = session.run([self._metric.values()[0]])
                    losses[eval_step] = metric_res
                    eval_step += 1
            if self._early_stopper is not None:
                self._early_stopper.append(train_step, np.mean(losses))
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="loss", simple_value=np.mean(losses)),
            ])
            self._file_writer.add_summary(summary, global_step=train_step)
            self._file_writer.flush()
            coord.request_stop()
            coord.join(threads)

    def _should_stop(self):
        """
        Assess whether the training should stop or not.
        :return: True if should stop False otherwise
        """
        return False if self._early_stopper is None else self._early_stopper.should_stop()
