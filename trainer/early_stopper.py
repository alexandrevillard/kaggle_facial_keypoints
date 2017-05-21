
class SimpleEarlyStopper(object):
    """
    Class to handle early stopping.
    """

    def __init__(self, chances=1):
        """
        :param chances: the number of chances the training has to beat the previously measured loss
        """
        if chances < 1:
            raise ValueError('Chances should be >= 1, was {}'.format(chances))
        self._chances = chances
        self._history = []
        self._best = None

    def append(self, g_step, loss):
        """
        Save the current loss, updates the best checkpoint if needed
        :param g_step: current global training step
        :param loss: current loss
        """
        to_add = (g_step, loss)
        self._history.append(to_add)
        if self._best is None or to_add[1] < self._best[1]:
            self._best = to_add

    def should_stop(self):
        """
        Assess whether the training should stop or not
        
        :return: True if should stop False otherwise
        """
        if len(self._history) <= 1:
            return False
        ms = self._history[-1][1] > self._history[-2][1]
        if ms:
            self._chances -= 1
        return self._chances == 0

    @property
    def best(self):
        return self._best
