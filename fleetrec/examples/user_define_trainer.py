from fleetrec.trainer.trainer import Trainer


class UserDefineTrainer(Trainer):
    def __init__(self, config=None):
        Trainer.__init__(self, config)
