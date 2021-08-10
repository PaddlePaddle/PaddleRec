#encoding=utf-8

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
from net import SRGNN

class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.dict_size = self.config.get("hyper_parameters.sparse_feature_number")
        self.hidden_size = self.config.get("hyper_parameters.sparse_feature_dim")
        self.step = self.config.get("hyper_parameters.gnn_propogation_steps")
        self.learning_rate = self.config.get("hyper_parameters.optimizer.learning_rate")
        self.decay_rate = self.config.get("hyper_parameters.optimizer.decay_rate")
        self.l2 = self.config.get("hyper_parameters.optimizer.l2")
        self.max_seq_len = self.config.get("runner.max_seq_len")
        self.max_uniq_len = self.config.get("runner.max_uniq_len")
        self.train_batch_size = self.config.get("runner.train_batch_size")
        self.test_batch_size = self.config.get("runner.infer_batch_size")


    def create_feeds(self,is_infer=None):
        '''
        items=np.array(items).astype("int64")
        seq_index=np.array(seq_index).astype("int32").reshape(-1,1)
        last_index = np.array([last_index]).astype("int32")
        adj_in = np.array(adj_in).astype("float32")
        adj_out = np.array(adj_out).astype("float32")
        mask = np.array(mask).astype("float32").reshape(-1, 1)
        label = np.array([label]).astype("int64")
        '''
        items=paddle.static.data(
            name="items",
            shape=[None,self.max_uniq_len],
            dtype="int64")
        seq_index=paddle.static.data(
            name="seq_index",
            shape=[None,self.max_seq_len,1],
            dtype="int32")
        last_index=paddle.static.data(
            name="last_index",
            shape=[None,1],
            dtype="int32")
        adj_in=paddle.static.data(
            name="adj_in",
            shape=[None,self.max_uniq_len,self.max_uniq_len],
            dtype="float32")
        adj_out=paddle.static.data(
            name="adj_out",
            shape=[None,self.max_uniq_len,self.max_uniq_len],
            dtype="float32")
        mask=paddle.static.data(
            name="mask",
            shape=[None,self.max_seq_len,1],
            dtype="float32")
        label=paddle.static.data(
            name="label",
            shape=[None,1],
            dtype="int64")
        feeds_list = [items, seq_index, last_index, adj_in, adj_out, mask, label]
        return feeds_list

    def net(self, inputs, is_infer=False):
        self.label_input = inputs[6]

        bs=self.train_batch_size if not is_infer else self.test_batch_size
        gnn_model = SRGNN(self.dict_size,self.hidden_size,self.step,batch_size=bs)

        probs = gnn_model(inputs)
        # auc=paddle.metric.Accuracy(topk=(20,))
        # correct = auc.compute(probs, inputs[6])
        # auc.update(correct)
        # res=auc.accumulate()
        res=paddle.static.accuracy(input=probs, label=self.label_input, k=20)

        self.inference_target_var = res
        if is_infer:
            fetch_dict = {'acc': res}
            return fetch_dict
        else:
            cost = F.cross_entropy(input=probs, label=self.label_input)
            self._cost = cost
            fetch_dict = {'cost': cost, 'acc': res}
            return fetch_dict

    def create_optimizer(self, strategy=None):
        scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=self.learning_rate,
            gamma=self.decay_rate,
            verbose=True)
        optimizer = paddle.optimizer.Adam(
            learning_rate=scheduler,
            weight_decay=L2Decay(self.l2))
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
