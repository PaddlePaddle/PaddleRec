import abc


class Model(object):
    """R
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        """R
        """
        self._cost = None
        self._metrics = {}
        self._data_var = []
        self._data_loader = None
        self._fetch_interval = 20
        self._namespace = "train.model"

    def get_inputs(self):
        return self._data_var

    def get_cost_op(self):
        """R
        """
        return self._cost

    def get_metrics(self):
        """R
        """
        return self._metrics

    def get_fetch_period(self):
        return self._fetch_interval

    @abc.abstractmethod
    def train_net(self):
        """R
        """
        pass

    def infer_net(self):
        pass
