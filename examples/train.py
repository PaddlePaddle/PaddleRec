
import yaml
from .. trainer.factory import TrainerFactory

if __name__ == "__main__":

    with open('ctr-dnn_train.yaml', 'r') as rb:
        global_config = yaml.load(rb.read())

    trainer = TrainerFactory.craete(global_config)
    trainer.run()
