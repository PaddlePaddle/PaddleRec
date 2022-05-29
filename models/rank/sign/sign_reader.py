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

from __future__ import print_function
import numpy as np
from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        # is train
        self.pred_edges = config.get("hyper_parameters.pred_edges", 1)
        self.batch_size = config.get("runner.train_batch_size", 1024)
        self.file_list = file_list
        self.config = config

        self.num_nodes_list = []
        self.edges_list = []
        self.node_features_list = []
        self.sr_list = []
        self.label_list = []

        # Process file
        self.process()

    def process(self):
        """process file"""
        node, edge, label, sr_list = self.read_data()

        for i in range(len(node)):
            num_nodes = len(node[i])

            node_features = np.array(
                node[i], dtype='int32').reshape(len(node[i]), 1)

            edges = []
            for u, v in zip(edge[i][0], edge[i][1]):
                u_v = (u, v)
                edges.append(u_v)

            sr = sr_list[i] if self.pred_edges else []

            self.num_nodes_list.append(num_nodes)
            self.edges_list.append(edges)
            self.node_features_list.append(node_features)
            self.sr_list.append(sr)
            self.label_list.append(label[i])

    def read_data(self):
        """read data"""
        node_list = []
        label = []
        data_num = 0  # number of datasets
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    data_num += 1
                    data = line.split()
                    # the first number is label
                    label.append(float(data[0]))
                    # the others is nodes
                    int_list = [int(data[i]) for i in range(len(data))[1:]]
                    node_list.append(int_list)

        if not self.pred_edges:
            edge_list = [[[], []] for _ in range(data_num)]
            sr_list = []
            # handle edges
            with open(self.edgefile, 'r') as f:
                for line in f:
                    edge_info = line.split()
                    node_index = int(edge_info[0])
                    edge_list[node_index][0].append(int(edge_info[1]))
                    edge_list[node_index][1].append(int(edge_info[2]))
        else:
            edge_list = []
            sr_list = []
            for index, nodes in enumerate(node_list):
                # for nodes in node_list:
                edge_l, sr_l = self.construct_full_edge_list(nodes)
                edge_list.append(edge_l)
                sr_list.append(sr_l)
        # Convert label to onehot encoding
        label = self.construct_one_hot_label(label)
        return node_list, edge_list, label, sr_list

    def construct_full_edge_list(self, nodes):
        num_node = len(nodes)
        edge_list = [[], []]  # [[sender...], [receiver...]]
        sender_receiver_list = []  # [[s,r],[s,r]...]
        for i in range(num_node):
            for j in range(num_node)[i:]:
                edge_list[0].append(i)
                edge_list[1].append(j)
                sender_receiver_list.append([nodes[i], nodes[j]])
        return edge_list, sender_receiver_list

    def construct_one_hot_label(self, label):
        """
        Convert label to onehot encoding
        input:[0,1,0,1]
        output:[[1,0] [0,1] [1,0] [0,1]]
        """
        nb_classes = int(max(label)) + 1
        targets = np.array(label, dtype=np.int32).reshape(-1)
        return np.eye(nb_classes)[targets]

    def __iter__(self):
        for i in range(len(self.label_list)):
            output_list = []
            output_list.append(np.array(self.num_nodes_list[i]))
            output_list.append(np.array(self.edges_list[i]))
            output_list.append(np.array(self.node_features_list[i]))
            output_list.append(np.array(self.sr_list[i]))
            output_list.append(np.array(self.label_list[i]))
            yield output_list
