import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
from net import SRGNN
from paddle.metric import Metric

class RecallK(Metric):
    def __init__(self,k,name='recallk',*args, **kwargs):
        super(RecallK, self).__init__(*args, **kwargs)
        self.topk=k
        self._name = name
        self.total=0
        self.count=0

    def compute(self,pred, label):
        # sort prediction and slice the top-5 scores
        pred = paddle.argsort(pred, descending=True)[:, :self.topk]
        # calculate whether the predictions are correct
        correct = pred == label
        return paddle.cast(correct, dtype='float32')

    def update(self, correct):
        accs = []
        num_corrects = correct[:, :self.topk].sum()
        num_samples = len(correct)
        accs.append(float(num_corrects) / num_samples)
        self.total += num_corrects
        self.count += num_samples
        return accs

    def reset(self):
        self.total = 0
        self.count = 0

    def accumulate(self):
        return float(self.total) / self.count if self.count != 0 else .0

    def name(self):
        return self._name

class DygraphModel():
    # define model
    def create_model(self, config):
        dict_size = config.get("hyper_parameters.sparse_feature_number")
        hidden_size = config.get("hyper_parameters.sparse_feature_dim")
        step = config.get("hyper_parameters.gnn_propogation_steps")

        gnn_model = SRGNN(dict_size,hidden_size,step)
        return gnn_model

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        tensor_data=[]
        for b in batch_data:
            tensor_data.append(paddle.to_tensor(b))
        return tensor_data

    # define loss function by predicts and label
    def create_loss(self, probs, label):
        cost = F.cross_entropy(input=probs, label=label)
        return cost

    # define optimizer
    def create_optimizer(self, dy_model, config):
        learning_rate = config.get("hyper_parameters.optimizer.learning_rate")
        decay_rate = config.get("hyper_parameters.optimizer.decay_rate")
        l2 = config.get("hyper_parameters.optimizer.l2")
        scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=learning_rate,
            gamma=decay_rate,
            verbose=True)
        optimizer = paddle.optimizer.Adam(
            learning_rate=scheduler,
            parameters=dy_model.parameters(),
            weight_decay=L2Decay(l2)
        )
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["recallk"]
        # recallk = paddle.metric.Accuracy(topk=(20,))
        recallk=RecallK(k=20)
        metrics_list = [recallk]
        return metrics_list, metrics_list_name

    # construct train forward phase
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        inputs = self.create_feeds(batch_data,config)

        probs = dy_model.forward(inputs)
        loss = self.create_loss(probs, inputs[6])
        # update metrics
        correct=metrics_list[0].compute(probs, inputs[6])
        metrics_list[0].update(correct)
        # print(metrics_list[0].total,metrics_list[0].count)

        # print_dict format :{'loss': loss}
        print_dict = None
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        inputs = self.create_feeds(batch_data,config)

        probs = dy_model.forward(inputs)
        # update metrics
        correct = metrics_list[0].compute(probs, inputs[6])
        metrics_list[0].update(correct)
        return metrics_list, None