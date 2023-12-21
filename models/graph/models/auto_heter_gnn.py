# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Implement a module that helps automatically generate heterogenous GNN
"""
import os
import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from collections import OrderedDict
import paddle.static as static
import pgl

__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append('../../../tools')
from utils.static_ps import model_util
from . import layers


class FeatureInteraction(nn.Layer):
    """ FeatureInteraction helps you to intergate features between relation
    """

    def __init__(self, interact_mode, hidden_size, etype_len):
        super(FeatureInteraction, self).__init__()
        self.interact_mode = interact_mode
        self.hidden_size = hidden_size
        self.etype_len = etype_len
        if etype_len > 1 and self.interact_mode.upper() == "GATNE":
            self.lin1 = nn.Linear(hidden_size, hidden_size, bias_attr=None)
            self.lin2 = nn.Linear(hidden_size, 1)

    def forward(self, feature_list):
        """ forward for feature interaction
        """
        if len(feature_list) == 1:
            return feature_list[0]
        elif self.interact_mode.upper() == "GATNE":
            U = paddle.stack(feature_list, axis=1)
            feature = F.tanh(self.lin1(U))
            alpha = self.lin2(feature).reshape([-1, len(feature_list)])
            alpha = F.softmax(alpha).unsqueeze([1])
            out = paddle.matmul(alpha, U).squeeze([1])
            return out
        else:
            return paddle.add_n(feature_list)

class Identity(nn.Layer):
    """ Identity function is use to return identity"""
    def __init__(self, hidden_size, act, use_degree_norm):
        super(Identity, self).__init__()

    def forward(self, x, degree_norm=None):
        """forward function 
        """
        return x
        


class AutoHeterGNN(nn.Layer):
    """AutoHeterGNN"""

    def __init__(self,
                 hidden_size,
                 layer_type,
                 num_layers,
                 etype_len,
                 act=None,
                 alpha_residual=0.9,
                 use_degree_norm=False,
                 interact_mode="sum",
                 return_weight=False):
        super(AutoHeterGNN, self).__init__()
        self.etype_len = etype_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layer_type = layer_type
        if self.layer_type == "Gatne":
            interact_mode = "gatne"
        self.alpha_residual = alpha_residual
        self.use_degree_norm = use_degree_norm
        self.return_weight = return_weight
        sub_layer_dict = OrderedDict()
        shared_sub_pre_layer = []

        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                cur_act = None
            else:
                cur_act = act
            shared_sub_pre_layer.append(getattr(layers, "Pre" + layer_type, Identity)(
                    hidden_size, act, use_degree_norm=use_degree_norm))
            for j in range(self.etype_len):
                sub_layer_dict[(i, j)] = getattr(layers, layer_type)(
                    hidden_size, act, use_degree_norm=use_degree_norm)
            sub_layer_dict[(i, self.etype_len)] = FeatureInteraction(
                interact_mode, hidden_size, self.etype_len)

        self.shared_sub_pre_layer = nn.LayerList(shared_sub_pre_layer)
        self.rgnn_dict = nn.LayerDict(sub_layer_dict)
        self.hcl_buffer = []

    def forward(self, graph_holders, init_feature, degree_norm=None):
        """ Forward for auto heter gnn
        """
        # pad a zeros to prevent empty graph happen
        zeros_tensor1 = paddle.zeros([1, init_feature.shape[-1]])
        zeros_tensor2 = paddle.zeros([1, 1], dtype="int64")
        init_feature = paddle.concat([zeros_tensor1, init_feature])
        feature = init_feature
        self.hcl_buffer.append(feature)

        if degree_norm is not None:
            degree_norm = degree_norm.reshape([self.etype_len, -1]).T
            degree_norm.stop_gradient = False
            degree_norm = paddle.sum(degree_norm, -1)
            degree_norm = model_util.get_degree_norm(degree_norm)
            degree_norm = paddle.concat([paddle.ones([1, 1], dtype="float32"), degree_norm], axis=0)

        for i in range(self.num_layers):
            graph_holder = graph_holders[self.num_layers - i - 1]
            num_nodes = graph_holder[0] + 1
            next_num_nodes = graph_holder[1] + 1
            edges_src = graph_holder[2] + 1
            edges_dst = graph_holder[3] + 1
            split_edges = graph_holder[4]
            # if self.return_weight:
            #     edges_weight = graph_holder[5]
            
            nxt_fs = []
            feature = self.shared_sub_pre_layer[i](feature, degree_norm)
            self.hcl_buffer.append(feature)

            for j in range(self.etype_len):
                start = paddle.zeros(
                    [1], dtype="int64") if j == 0 else split_edges[j - 1]
                new_edges_src = paddle.concat(
                    [zeros_tensor2, edges_src[start:split_edges[j]]])
                new_edges_dst = paddle.concat(
                    [zeros_tensor2, edges_dst[start:split_edges[j]]])
                graph = pgl.Graph(
                    num_nodes=num_nodes,
                    edges=paddle.concat(
                        [new_edges_src, new_edges_dst], axis=1))

                # generate feature of single relation
                nxt_f = self.rgnn_dict[(i, j)](graph, feature, next_num_nodes, degree_norm)
                nxt_fs.append(nxt_f)
            # feature intergation
            feature = self.rgnn_dict[(i, self.etype_len)](nxt_fs)

            # heter graph residual
            feature = init_feature[:
                                   next_num_nodes] * self.alpha_residual + feature * (
                                       1 - self.alpha_residual)
            if degree_norm is not None:
                degree_norm = degree_norm[:next_num_nodes]
        # remove first zeros to prevent empty graph happen
        return feature[1:]
