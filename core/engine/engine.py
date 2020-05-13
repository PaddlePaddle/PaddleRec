import abc


class Engine:
    __metaclass__ = abc.ABCMeta

    def __init__(self, envs, trainer):
        self.envs = envs
        self.trainer = trainer
        self.__init_impl__()

    def __init_impl__(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

