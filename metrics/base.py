"""
Do metric jobs. calculate AUC, MSE, COCP ...
"""
import abc


class Metric(object):
    """R
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        """ """
        pass

    @abc.abstractmethod
    def clear(self, scope, params):
        """
        clear current value
        Args:
            scope: value container
            params: extend varilable for clear
        """
        pass

    @abc.abstractmethod
    def calculate(self, scope, params):
        """
        calculate result
        Args:
            scope: value container
            params: extend varilable for clear
        """
        pass

    @abc.abstractmethod
    def get_result(self):
        """
        Return:
            result(dict) : calculate result
        """
        pass

    @abc.abstractmethod
    def get_result_to_string(self):
        """
        Return:
            result(string) : calculate result with string format, for output
        """
        pass
