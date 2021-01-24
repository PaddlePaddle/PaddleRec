#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from net import TextCNNLayer


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.dict_dim = self.config.get("hyper_parameters.dict_dim")
        self.max_len = self.config.get("hyper_parameters.max_len")
        self.cnn_dim = self.config.get("hyper_parameters.cnn_dim")
        self.cnn_filter_size1 = self.config.get(
            "hyper_parameters.cnn_filter_size1")
        self.cnn_filter_size2 = self.config.get(
            "hyper_parameters.cnn_filter_size2")
        self.cnn_filter_size3 = self.config.get(
            "hyper_parameters.cnn_filter_size3")
        self.filter_sizes = [
            self.cnn_filter_size1, self.cnn_filter_size2, self.cnn_filter_size3
        ]
        self.emb_dim = self.config.get("hyper_parameters.emb_dim")
        self.hid_dim = self.config.get("hyper_parameters.hid_dim")
        self.class_dim = self.config.get("hyper_parameters.class_dim")
        self.is_sparse = self.config.get("hyper_parameters.is_sparse")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")

    def create_feeds(self, is_infer=False):
        data = paddle.static.data(
            name="input", shape=[None, self.max_len], dtype='int64')
        label = paddle.static.data(
            name="label", shape=[None, 1], dtype='int64')
        return [data, label]

    def net(self, input, is_infer=False):
        """ network definition """
        data = input[0]
        label = input[1]

        textcnn_model = TextCNNLayer(
            self.dict_dim,
            self.emb_dim,
            self.class_dim,
            cnn_dim=self.cnn_dim,
            filter_sizes=self.filter_sizes,
            hidden_size=self.hid_dim)

        pred = textcnn_model.forward(data)
        # softmax layer
        prediction = paddle.nn.functional.softmax(pred)

        acc = paddle.metric.accuracy(input=prediction, label=label)
        if is_infer:
            fetch_dict = {'acc': acc}
            return fetch_dict
        cost = paddle.nn.functional.cross_entropy(input=pred, label=label)
        avg_cost = paddle.mean(x=cost)

        self._cost = avg_cost
        fetch_dict = {'cost': avg_cost, 'acc': acc}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        optimizer = paddle.optimizer.Adam(
            learning_rate=self.learning_rate, lazy_mode=True)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
