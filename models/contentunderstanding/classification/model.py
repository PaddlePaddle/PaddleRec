import paddle.fluid as fluid
import math

from paddlerec.core.utils import envs
from paddlerec.core.model import Model as ModelBase

import paddle.fluid as fluid
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.tensor as tensor
import paddle.fluid.layers.control_flow as cf

class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)
        self.dict_dim = 100
        self.max_len = 10
        self.cnn_dim = 32
        self.cnn_filter_size = 128
        self.emb_dim = 8
        self.hid_dim = 128
        self.class_dim = 2

    def train_net(self):
        """ network definition """
       
        data = fluid.data(name="input", shape=[None, self.max_len], dtype='int64')
        label = fluid.data(name="label", shape=[None, 1], dtype='int64')
        seq_len = fluid.data(name="seq_len", shape=[None], dtype='int64')
        # embedding layer
        emb = fluid.embedding(input=data, size=[self.dict_dim, self.emb_dim])
        emb = fluid.layers.sequence_unpad(emb, length=self.seq_len)
        # convolution layer
        conv = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=self.cnn_dim,
            filter_size=self.cnn_filter_size,
            act="tanh",
            pool_type="max")

        # full connect layer
        fc_1 = fluid.layers.fc(input=[conv], size=hid_dim)
        # softmax layer
        prediction = fluid.layers.fc(input=[fc_1], size=self.class_dim, act="softmax")
        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc = fluid.layers.accuracy(input=prediction, label=label) 

        self.cost = avg_cost
        self.metrics["acc"] = cos_pos

    def get_cost_op(self):
        return self.cost

    def get_metrics(self):
        return self.metrics

    def optimizer(self):
        learning_rate = 0.01
        sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=learning_rate)
        return sgd_optimizer

    def infer_net(self, parameter_list):
        self.train_net()
