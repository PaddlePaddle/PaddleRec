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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
import numpy as np
from graph import Graph


class L0_SIGN(nn.Layer):
    def __init__(self, pred_edges, n_feature, dim, hidden_layer, l0_para,
                 batch_size):
        super(L0_SIGN, self).__init__()

        self.pred_edges = pred_edges
        self.n_feature = n_feature
        self.dim = dim
        self.hidden_layer = hidden_layer
        self.l0_para = l0_para
        self.batch_size = batch_size

        self.linkpred = LinkPred(self.dim, self.hidden_layer, self.n_feature,
                                 self.l0_para)
        self.sign = SIGN(self.dim, self.hidden_layer)
        self.g = paddle.nn.Linear(self.dim, 2)
        self.feature_emb = nn.Embedding(self.n_feature, self.dim)

    def forward(self,
                edges,
                node_feat,
                edge_feat,
                segment_ids,
                is_training=True):
        # does not conduct link prediction, use all interactions
        # graph: pgl.Graph object
        # graph.node_feat['node_attr']: [bacth_size*3, 1]
        # graph.edge_feat['edge_attr']: [bact_size*6, 2]
        # graph.edges: [bact_size*6, 2]

        graph = Graph(
            edges=edges,
            node_feat={"node_attr": node_feat},
            edge_feat={"edge_attr": edge_feat}).tensor()

        x, edge_index, sr = graph.node_feat[
            'node_attr'], graph.edges, graph.edge_feat['edge_attr']
        _x = self.feature_emb(paddle.cast(x, 'int32'))
        _x = _x.squeeze(1)

        graph.node_feat['node_attr'] = _x

        if self.pred_edges:
            sr = paddle.transpose(sr, perm=[1, 0])
            s, l0_penaty = self.linkpred(sr, is_training)
            pred_edge_index, pred_edge_weight = self.construct_pred_edge(
                edge_index, s)

            sub_graph = Graph(
                edges=pred_edge_index, node_feat={'node_attr': _x})
            updated_nodes = self.sign(sub_graph, pred_edge_weight)
        else:
            updated_nodes = self.sign(graph, sr)
            l0_penaty = 0

        l2_penaty = paddle.multiply(updated_nodes, updated_nodes).sum()

        while updated_nodes.shape[0] < segment_ids.shape[0]:
            updated_nodes = paddle.concat(
                [
                    updated_nodes, paddle.to_tensor(
                        paddle.zeros((1, self.dim)), dtype='float32')
                ],
                0)

        # Add graph-average-pooling
        graph_embedding = pgl.math.segment_mean(updated_nodes, segment_ids)
        out = self.g(graph_embedding)
        out = paddle.clip(out, min=0, max=1)

        return out, l0_penaty, l2_penaty

    def construct_pred_edge(self, fe_index, s):
        """
        fe_index: full_edge_index, [2, all_edges_batchwise]
        s: predicted edge value, [all_edges_batchwise, 1]

        construct the predicted edge set and corresponding edge weights
        """
        s = s[:, 0]

        fe_index = paddle.transpose(fe_index, perm=[1, 0])

        sender = paddle.unsqueeze(fe_index[0][s > 0], 0)
        receiver = paddle.unsqueeze(fe_index[1][s > 0], 0)

        pred_index = paddle.concat([sender, receiver], 0)
        pred_weight = s[s > 0]
        pred_index = paddle.transpose(pred_index, perm=[1, 0])

        return pred_index, pred_weight


class SIGN(nn.Layer):
    def __init__(self, dim, hidden_layer, aggr_func="mean"):
        super(SIGN, self).__init__()
        assert aggr_func in [
            "sum", "mean", "max", "min"
        ], "Only support 'sum', 'mean', 'max', 'min' built-in receive function."
        self.aggr_func = "reduce_%s" % aggr_func

        ## Sets the initialization weight
        self.lin1 = nn.Linear(
            dim,
            hidden_layer,
            weight_attr=paddle.nn.initializer.KaimingUniform())
        self.lin2 = nn.Linear(
            hidden_layer,
            dim,
            weight_attr=paddle.nn.initializer.KaimingUniform())
        self.activation = nn.ReLU()

    def _send_func(self, src_feat, dst_feat, edge_feat=None):
        pairwise_analysis = self.lin1(
            paddle.multiply(src_feat["src"], dst_feat["dst"]))
        pairwise_analysis = self.activation(pairwise_analysis)
        pairwise_analysis = self.lin2(pairwise_analysis)

        if edge_feat != None:
            edge_feat_ = paddle.reshape(edge_feat["e_attr"], [-1, 1])
            interaction_analysis = paddle.multiply(pairwise_analysis,
                                                   edge_feat_)
        else:
            interaction_analysis = pairwise_analysis

        return {"msg": interaction_analysis}

    def _recv_func(self, msg):
        return getattr(msg, self.aggr_func)(msg["msg"])

    def forward(self, graph, edge_attr):
        msg = graph.send(
            self._send_func,
            src_feat={"src": graph.node_feat['node_attr'].clone()},
            dst_feat={"dst": graph.node_feat['node_attr'].clone()},
            edge_feat={"e_attr": edge_attr})
        output = graph.recv(self._recv_func, msg)

        return output


class LinkPred(nn.Layer):
    def __init__(self, D_in, H, n_feature, l0_para):
        super(LinkPred, self).__init__()
        ## Sets the initialization weight
        self.linear1 = nn.Linear(
            D_in, H, weight_attr=nn.initializer.KaimingUniform())
        self.linear2 = nn.Linear(
            H, 1, weight_attr=nn.initializer.KaimingUniform())
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        with paddle.no_grad():
            self.linear2.weight.set_value(self.linear2.weight + 0.2)

        self.temp = l0_para[0]
        self.inter_min = l0_para[1]
        self.inter_max = l0_para[2]

        self.feature_emb_edge = nn.Embedding(
            n_feature,
            D_in,
            weight_attr=paddle.ParamAttr(
                name='emb_weight',
                initializer=nn.initializer.Normal(
                    mean=0.2, std=0.01)))

    def forward(self, sender_receiver, is_training):
        # Construct permutation input
        sender_emb = self.feature_emb_edge(
            paddle.cast(sender_receiver[0, :], 'int32'))
        receiver_emb = self.feature_emb_edge(
            paddle.cast(sender_receiver[1, :], 'int32'))

        _input = paddle.multiply(sender_emb, receiver_emb)
        h_relu = self.dropout(self.relu(self.linear1(_input)))
        loc = self.linear2(h_relu)
        if is_training:
            u = paddle.rand(loc.shape, dtype=loc.dtype)
            u.stop_gradient = False
            logu = paddle.log2(u)
            logmu = paddle.log2(1 - u)
            sum_log = loc + logu - logmu
            s = F.sigmoid(sum_log / self.temp)
            s = s * (self.inter_max - self.inter_min) + self.inter_min
        else:
            s = F.sigmoid(loc) * (self.inter_max - self.inter_min
                                  ) + self.inter_min

        s = paddle.clip(s, min=0, max=1)

        l0_penaty = F.sigmoid(loc - self.temp * np.log2(-self.inter_min /
                                                        self.inter_max)).mean()

        return s, l0_penaty
