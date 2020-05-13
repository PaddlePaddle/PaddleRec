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

    def train_net(self):
        """ network definition """
       
        data = fluid.data(name="input", shape=[None, max_len], dtype='int64')
        label = fluid.data(name="label", shape=[None, 1], dtype='int64')
        seq_len = fluid.data(name="seq_len", shape=[None], dtype='int64')
        # embedding layer
        emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
        emb = fluid.layers.sequence_unpad(emb, length=seq_len)
        # convolution layer
        conv = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=cnn_dim,
            filter_size=cnn_filter_size,
            act="tanh",
            pool_type="max")

        # full connect layer
        fc_1 = fluid.layers.fc(input=[conv], size=hid_dim)
        # softmax layer
        prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")
        #if is_prediction:
        #    return prediction
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
        learning_rate = 0.01#envs.get_global_env("hyper_parameters.base_lr", None, self._namespace)
        sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=learning_rate)
        #sgd_optimizer.minimize(avg_cost)
        return sgd_optimizer


    def infer_net(self, parameter_list):
        self.train_net()
