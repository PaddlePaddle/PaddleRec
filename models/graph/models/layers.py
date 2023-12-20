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
"""Implemented all gnn layers
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
import os
import sys
import math

__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append('../../../tools')
from utils.static_ps import model_util


class GIN(nn.Layer):
    """GIN"""

    def __init__(self, hidden_size, act, use_degree_norm=False):
        super(GIN, self).__init__()
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.act = act
        self.use_degree_norm = use_degree_norm

    def forward(self, graph, x, next_num_nodes, degree_norm=None):
        """ forward function
        """
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        self_feature = x[:next_num_nodes]
        
        if self.use_degree_norm:
            x = x * degree_norm

        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, "sum", out_size=next_num_nodes)

        if self.use_degree_norm:
            sub_degree_norm = model_util.get_graph_degree_norm(graph)[:next_num_nodes]
            neigh_feature *= sub_degree_norm 

        output = self_feature + neigh_feature
        output = self.lin(output)
        if self.act is not None:
            output = getattr(F, self.act)(output)
        output = output + self_feature
        return output


class GraphSAGEMean(nn.Layer):
    """GraphSAGEMean"""

    def __init__(self, hidden_size, act, use_degree_norm=False):
        super(GraphSAGEMean, self).__init__()
        self.lin = nn.Linear(2 * hidden_size, hidden_size)
        self.act = act
        self.use_degree_norm = use_degree_norm
        if self.use_degree_norm:
            raise ValueError("degree norm is not allowed for GraphSAGEMean.")

    def forward(self, graph, x, next_num_nodes, degree_norm=None):
        """ forward function
        """
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, "mean", out_size=next_num_nodes)
        self_feature = x[:next_num_nodes]
        output = paddle.concat([self_feature, neigh_feature], axis=1)
        output = self.lin(output)
        if self.act is not None:
            output = getattr(F, self.act)(output)
        output = F.normalize(output, axis=-1)
        return output


class GraphSAGEBow(nn.Layer):
    """GraphSAGEBow"""

    def __init__(self, hidden_size, act, use_degree_norm=False):
        super(GraphSAGEBow, self).__init__()
        self.use_degree_norm = use_degree_norm
        if self.use_degree_norm:
            raise ValueError("degree norm is not allowed for GraphSAGEBow.")

    def forward(self, graph, x, next_num_nodes, degree_norm=None):
        """ forward function
        """
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, "mean", out_size=next_num_nodes)
        self_feature = x[:next_num_nodes]
        output = self_feature + neigh_feature
        output = F.normalize(output, axis=-1)
        return output


class GraphSAGEMax(nn.Layer):
    """GraphSAGEMax"""

    def __init__(self, hidden_size, act, use_degree_norm=False):
        super(GraphSAGEMax, self).__init__()
        self.lin = nn.Linear(2 * hidden_size, hidden_size)
        self.act = act
        if self.use_degree_norm:
            raise ValueError("degree norm is not allowed for GraphSAGEMax.")

    def forward(self, graph, x, next_num_nodes, degree_norm=None):
        """ forward function
        """
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, "max", out_size=next_num_nodes)
        self_feature = x[:next_num_nodes]
        output = paddle.concat([self_feature, neigh_feature], axis=1)
        output = self.lin(output)
        if self.act is not None:
            output = getattr(F, self.act)(output)
        output = F.normalize(output, axis=-1)
        return output


class GAT(nn.Layer):
    """GAT"""

    def __init__(self, hidden_size, act, use_degree_norm=False):
        super(GAT, self).__init__()
        self.gnn = pgl.nn.GATConv(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_heads=1,
            feat_drop=0,
            attn_drop=0,
            activation=act)
        self.lin = nn.Linear(hidden_size * 2, hidden_size)
        self.act = act
        self.use_degree_norm = use_degree_norm

    def forward(self, graph, x, next_num_nodes, degree_norm=None):
        """ forward function
        """
        self_feature = x[:next_num_nodes]
        if self.use_degree_norm:
            x = x * degree_norm
        neigh_feature = self.gnn(graph, x)[:next_num_nodes]

        if self.use_degree_norm:
            sub_degree_norm = model_util.get_graph_degree_norm(graph)[:next_num_nodes]
            neigh_feature *= sub_degree_norm

        output = self.lin(paddle.concat([self_feature, neigh_feature], axis=1))
        if self.act is not None:
            output = getattr(F, self.act)(output)
        return output


class LightGCN(nn.Layer):
    """LightGCN"""

    def __init__(self, hidden_size, act, use_degree_norm=False):
        super(LightGCN, self).__init__()
        self.use_degree_norm = use_degree_norm

    def forward(self, graph, x, next_num_nodes, degree_norm=None):
        """ forward function
        """
        src, dst = graph.edges[:, 0], graph.edges[:, 1]
        if self.use_degree_norm:
            x = x * degree_norm

        neigh_feature = paddle.geometric.send_u_recv(
            x, src, dst, "sum", out_size=next_num_nodes)

        if self.use_degree_norm:
            sub_degree_norm = model_util.get_graph_degree_norm(graph)[:next_num_nodes]
            neigh_feature *= sub_degree_norm

        return neigh_feature

class TransformerConv(nn.Layer):
    """TransformerConv"""

    def __init__(self, hidden_size, act, use_degree_norm=False):
        super(TransformerConv, self).__init__()
        self.hidden_size = hidden_size 
        self.nheads = 4
        self.head_dim = self.hidden_size // self.nheads
        self.act = act
        self.use_degree_norm = use_degree_norm
        self.edge_type = self.create_parameter(shape=[1, self.nheads, self.head_dim], is_bias=True) 
        self.gate = self.create_parameter(shape=[1, self.nheads])

    def send_attention(self, src_feat, dst_feat, edge_feat):
        """send attention"""
        alpha = dst_feat["q"] * src_feat["k"]
        alpha = paddle.sum(alpha, axis=-1)
        if self.use_degree_norm:
            return {"alpha": alpha, "v": src_feat["v"], "deg": src_feat["deg"] * dst_feat["deg"]}
        else:
            return {"alpha": alpha, "v": src_feat["v"]} 

    def reduce_attention(self, msg):
        """reduce attention"""
        if self.use_degree_norm:
            gate = F.sigmoid(self.gate) 
            alpha = gate * msg.reduce_softmax(msg["alpha"]) + (1 - gate) * msg["deg"]
        else:
            alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.nheads, 1])
        feature = msg["v"]
        feature = feature * alpha
        feature = paddle.reshape(feature,
                                     [-1, self.nheads * self.head_dim])
        feature = msg.reduce(feature, pool_type="sum")
        return feature 

    def send_recv(self, graph, q, k, v, degree_norm):
        """send recv"""
        if self.use_degree_norm:
            sub_degree_norm = model_util.get_graph_degree_norm(graph)
            msg = graph.send(
                self.send_attention,
                src_feat={"k": k, "v": v, "deg": degree_norm},
                dst_feat={"q": q, "deg": sub_degree_norm})
        else:
            msg = graph.send(
                self.send_attention,
                src_feat={"k": k,
                          "v": v},
                dst_feat={"q": q})
 

        output = graph.recv(reduce_func=self.reduce_attention, msg=msg)
        return output 

    def forward(self, graph, x, next_num_nodes, degree_norm=None):
        """ forward function
        """
        q, k, v, ori_x = x
        k = k + self.edge_type

        neigh_feature = self.send_recv(graph, q=q, k=k, v=v, degree_norm=degree_norm)
        return neigh_feature[:next_num_nodes]


class PreTransformerConv(nn.Layer):
    """PreTransformerConv"""

    def __init__(self, hidden_size, act, use_degree_norm=False):
        super(PreTransformerConv, self).__init__()
        self.hidden_size = hidden_size 
        self.nheads = 4
        self.head_dim = self.hidden_size // self.nheads
        self.act = act
        self.use_degree_norm = use_degree_norm
        self.q = self.linear(self.hidden_size, self.hidden_size)
        self.k = self.linear(self.hidden_size, self.hidden_size)
        self.v = self.linear(self.hidden_size, self.hidden_size)

    def linear(self, input_size, hidden_size, beta=0.1):
        """ Init Linear with less scale"""
        fan_in = input_size
        bias_bound = 1.0 / math.sqrt(fan_in) * beta
        fc_bias_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-bias_bound, high=bias_bound))

        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        weight_bound = weight_bound * beta 
        fc_w_attr = paddle.ParamAttr(initializer=nn.initializer.Uniform(low=-weight_bound, high=weight_bound))
        return nn.Linear(input_size, hidden_size, weight_attr=fc_w_attr, bias_attr=fc_bias_attr)
    

    def forward(self, x, degree_norm=None):
        """ forward function
        """
        ori_x = x
        if self.use_degree_norm:
            x = x * degree_norm
     
        q = self.q(x) + x 
        k = self.k(x) + x 
        v = self.v(ori_x) + ori_x
        q = paddle.reshape(q, [-1, self.nheads, self.head_dim])
        k = paddle.reshape(k, [-1, self.nheads, self.head_dim])
        v = paddle.reshape(v, [-1, self.nheads, self.head_dim])
        return (q, k, v, ori_x) 
