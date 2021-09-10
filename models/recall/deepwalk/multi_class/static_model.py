# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle
import pgl
import sys
import numpy as np
from sklearn.metrics import f1_score
sys.path.append("..")
from net import MultiClassLayer
from randwalk_reader import ShardedDataset, BatchRandWalk


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.epochs = self.config.get("runner.epochs", None)
        self.embed_size = self.config.get("hyper_parameters.embed_size", None)

    def create_feeds(self, is_infer=False):
        dataset = pgl.dataset.BlogCatalogDataset()
        indegree = dataset.graph.indegree()
        outdegree = dataset.graph.outdegree()
        self.num_nodes = dataset.graph.num_nodes
        self.num_groups = dataset.num_groups
        self.batch_size = len(dataset.train_index)  #5156
        if is_infer:
            self.all_nodes = dataset.test_index
        else:
            self.all_nodes = dataset.train_index
        self.node_group_id = dataset.graph.node_feat['group_id']

        node = paddle.static.data(name="node", shape=[None], dtype='int64')
        labels = paddle.static.data(
            name="labels", shape=[None, 39], dtype='float32')
        feeds_list = [node, labels]
        return feeds_list

    def create_data_loader(self):
        if self.all_nodes is None:
            self.all_nodes = np.arange(self.num_nodes)

        def batch_nodes_generator():
            perm = np.arange(len(self.all_nodes), dtype=np.int64)
            start = 0
            while start < len(self.all_nodes):
                yield self.all_nodes[perm[start:start + self.batch_size]]
                start += self.batch_size

        def wrapper():
            for batch_nodes in batch_nodes_generator():
                batch_labels = self.node_group_id[batch_nodes].astype(
                    np.float32)
                yield [batch_nodes, batch_labels]

        return wrapper

    def net(self, input, is_infer=False):
        node = input[0]
        labels = input[1]
        num_samples = self.batch_size
        multi_class_model = MultiClassLayer(self.num_nodes, self.embed_size,
                                            self.num_groups)
        logits = multi_class_model(node)
        probs = paddle.nn.functional.sigmoid(logits)
        bce_loss = paddle.nn.BCEWithLogitsLoss()
        loss = bce_loss(logits, labels)
        avg_cost = paddle.mean(x=loss)
        self._cost = avg_cost
        topk = labels.sum(-1)

        fetch_dict = {
            "cost": avg_cost,
            "labels": labels,
            "probs": probs,
            "topk": topk
        }
        return fetch_dict

    def create_optimizer(self, strategy=None):
        scheduler = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.epochs,
            end_lr=0.0001)
        optimizer = paddle.optimizer.Adam(learning_rate=scheduler)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)

    def infer_net(self, input):
        return self.net(input, is_infer=True)
