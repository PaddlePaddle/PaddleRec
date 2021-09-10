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
sys.path.append("..")
from net import SkipGramLayer
from randwalk_reader import ShardedDataset, BatchRandWalk


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.config = config
        self._init_hyper_parameters()

    def _init_hyper_parameters(self):
        self.neg_num = self.config.get("hyper_parameters.neg_num")
        self.walk_len = self.config.get("hyper_parameters.walk_len")
        self.win_size = self.config.get("hyper_parameters.win_size")
        self.neg_sample_type = self.config.get(
            "hyper_parameters.neg_sample_type")
        self.embed_size = self.config.get("hyper_parameters.embed_size")
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.sample_workers = self.config.get(
            "hyper_parameters.sample_workers")
        self.decay_steps = self.config.get("hyper_parameters.decay_steps")
        self.use_gpu = self.config.get("runner.use_gpu", True)
        self.step_epoch = self.config.get("hyper_parameters.step_epoch", None)

    def create_feeds(self, is_infer=False):
        dataset = pgl.dataset.BlogCatalogDataset()
        indegree = dataset.graph.indegree()
        outdegree = dataset.graph.outdegree()
        self.num_nodes = dataset.graph.num_nodes
        self.train_ds = ShardedDataset(
            dataset.graph.nodes, repeat=self.step_epoch)
        self.collate_fn = BatchRandWalk(dataset.graph, self.walk_len,
                                        self.win_size, self.neg_num,
                                        self.neg_sample_type)
        src = paddle.static.data(name="src", shape=[-1, 1], dtype="int64")
        dsts = paddle.static.data(
            name="dsts", shape=[-1, self.neg_num + 1], dtype="int64")
        feeds_list = [src, dsts]
        return feeds_list

    def create_data_loader(self):
        batch_size = self.config.get('runner.train_batch_size', None)
        data_loader = pgl.utils.data.Dataloader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.sample_workers,
            collate_fn=self.collate_fn)
        return data_loader

    def net(self, input, is_infer=False):
        src = input[0]
        dsts = input[1]
        skip_gramp_model = SkipGramLayer(
            self.num_nodes,
            self.embed_size,
            self.neg_num,
            sparse=not self.use_gpu)
        loss = skip_gramp_model(src, dsts)

        avg_cost = paddle.mean(x=loss)
        self._cost = avg_cost

        fetch_dict = {'cost': avg_cost}
        return fetch_dict

    def create_optimizer(self, strategy=None):
        scheduler = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.decay_steps,
            end_lr=0.0001)
        optimizer = paddle.optimizer.Adam(learning_rate=scheduler)
        if strategy != None:
            import paddle.distributed.fleet as fleet
            strategy.sharding = True
            strategy.sharding_configs = {
                "segment_anchors": None,
                "sharding_segment_strategy": "segment_broadcast_MB",
                "segment_broadcast_MB": 32,
                "sharding_degree": int(paddle.distributed.get_world_size()),
            }
            optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(self._cost)
