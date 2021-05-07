import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import os
from net import DeepRetrieval
from paddle.distributed.fleet.dataset.index_dataset import GraphIndex


class StaticModel():
    def __init__(self, config):
        self.cost = None
        self.infer_target_var = None
        self.config = config
        self._init_hyper_parameters()

    def save_item_path(self, path, prefix):
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, str(prefix))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, self.path_save_file_name)
        self.graph_index._graph.save(path)

    def load_item_path(self, path):
        path = os.path.join(path, self.path_save_file_name)
        self.graph_index._graph.load(path)

    def _init_hyper_parameters(self):
        config = self.config
        self.path_save_file_name = "path_save";
        self.width = config.get("hyper_parameters.width")
        self.height = config.get("hyper_parameters.height")
        self.beam_search_num = config.get(
            "hyper_parameters.beam_search_num")
        self.item_path_volume = config.get(
            "hyper_parameters.item_path_volume")
        self.user_embedding_size = config.get(
            "hyper_parameters.user_embedding_size")
        self.graph_index = GraphIndex(
            "test", self.width, self.height, self.item_path_volume)
        self.learning_rate = config.get("hyper_parameters.optimizer.learning_rate", 0.001)

        init_model_path = config.get("model_init_path")

        if init_model_path == None:
            self.graph_index._init_by_random()
        else:
            self.graph_index._init_graph(os.path.join(init_model_path, self.path_save_file_name))
        self.use_multi_task_learning = config.get("hyper_parameters.use_multi_task_learning")
        self.item_count = config.get("hyper_parameters.item_count")
        if self.use_multi_task_learning:
            self.multi_task_layer_size = config.get("hyper_parameters.multi_task_layer_size")
        else:
            self.multi_task_layer_size = None


    def create_feeds(self, is_infer=False):
        inputs = [
                        paddle.static.data(
                    name="item_id",
                    shape=[None, 1],
                    dtype='int32'),
            paddle.static.data(
                name="user_embedding_input",
                shape=[None, self.user_embedding_size],
                dtype='float32'), 
                paddle.static.data(
                    name="kd_path_input",
                    shape=[None, self.item_path_volume, self.width],
                    dtype='int32')
        ]
        if self.use_multi_task_learning:
            print("use multi_task_learning----")
            inputs.append(            paddle.static.data(
                name="multi_task_pos_label",
                shape=[None, 1],
                dtype='int32'))
            inputs.append(paddle.static.data(
                name="multi_task_neg_label",
                shape=[None, 1],
                dtype='int32'))
        return inputs
        
    def net(self, input, is_infer=False):
        self.user_embedding = input[1]
        self.kd_path = input[2]
        self.multi_task_pos_label = self.multi_task_neg_label = None
        if self.use_multi_task_learning:
            self.multi_task_pos_label = input[3]
            self.multi_task_neg_label = input[4]


        self.model = DeepRetrieval(self.width, self.height,
                          self.beam_search_num, self.item_path_volume,
                          self.user_embedding_size, self.item_count,
                          self.use_multi_task_learning, self.multi_task_layer_size, is_static = True)

        path_prob, multi_task_loss = self.model(self.user_embedding, self.kd_path, self.multi_task_pos_label, self.multi_task_neg_label)


        cost = self.create_loss(path_prob, multi_task_loss)
        avg_cost = paddle.mean(x=cost)
        self._cost = avg_cost

        fetch_dict = {'cost': avg_cost}
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

    def create_loss(self, path_prob, multi_task_loss):
        # path_prob: (batch_size * J, D)
        path_prob = paddle.prod(
            path_prob, axis=1, keepdim=True)  # (batch_size* J, 1)
        item_path_prob = paddle.reshape(
            path_prob, (-1, self.item_path_volume))  # (batch_size, J)
        item_path_prob = paddle.sum(item_path_prob, axis=1, keepdim=True)
        item_path_prob_log = paddle.log(item_path_prob)
        cost = -1 * paddle.sum(item_path_prob_log)
        print("cost: {}".format(cost))
        if self.use_multi_task_learning:
            cost = cost + multi_task_loss
        return cost

        # define optimizer