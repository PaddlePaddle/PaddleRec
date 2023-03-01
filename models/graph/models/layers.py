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
from . import model_util


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
            sub_degree_norm = model_util.get_graph_degree_norm(
                graph)[:next_num_nodes]
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
            sub_degree_norm = model_util.get_graph_degree_norm(
                graph)[:next_num_nodes]
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
            sub_degree_norm = model_util.get_graph_degree_norm(
                graph)[:next_num_nodes]
            neigh_feature *= sub_degree_norm

        return neigh_feature
