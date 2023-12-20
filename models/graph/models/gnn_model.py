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
"""
    gnn model.
"""
import os
import sys
import math
import time
import numpy as np
import pickle as pkl

import paddle
import paddle.incubate as F
import paddle.static as static
import pgl
from pgl.utils.logger import log

from . import layers
from . import loss as Loss

__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append('../../../tools')
from utils.static_ps import util
from utils.static_ps import model_util
from . import helper
from .auto_heter_gnn import AutoHeterGNN


class GNNModel(object):
    """ GNNModel """

    def __init__(self, config, dump_file_obj=None, is_predict=False):
        """ init """
        self.config = config
        self.dump_file_obj = dump_file_obj
        self.is_predict = is_predict
        self.neg_num = self.config.neg_num
        self.emb_size = self.config.emb_size
        self.hidden_size = self.emb_size
        self._use_cvm = False

        self.holder_list = []

        self.show_clk, holder_list = model_util.build_show_clk_holder(self.config, is_predict)
        self.holder_list.extend(holder_list)

        self.nodeid_slot_holder, holder_list = \
                model_util.build_node_holder(self.config.nodeid_slot, self.config, is_predict)
        self.holder_list.extend(holder_list)

        self.discrete_slot_holders, self.discrete_slot_lod_holders, holder_list = \
                model_util.build_slot_holder(self.config.slots)
        self.holder_list.extend(holder_list)

        # float slot holder
        self.float_slot_holder, self.float_slot_lod_holder, holder_list = \
                model_util.build_float_holder(self.config.float_slots, \
                                              self.config.float_slots_len)
        self.holder_list.extend(holder_list)

        tmp_holder = []
        for i in range(len(self.float_slot_holder)):
            float_holder = self.float_slot_holder[i]
            float_lod_holder = self.float_slot_lod_holder[i]
            float_slot = self.config.float_slots[i]
            float_slot_len = self.config.float_slots_len[str(float_slot)]
            float_holder = paddle.incubate.operators.unzip(
                  float_holder, float_lod_holder, int(float_slot_len))
            float_holder.stop_gradient = True
            float_holder = paddle.reshape(float_holder, [-1, int(float_slot_len)])
            tmp_holder.append(float_holder)
        self.float_slot_holder = tmp_holder

        if self.config.sage_mode:
            use_degree_norm = False if not self.config.use_degree_norm else True
            self.return_weight = False if not self.config.return_weight else True
            holder_dict = \
                    model_util.build_graph_holder(self.config.samples, use_degree_norm,
                                                  self.return_weight)
            self.graph_holders = holder_dict["graph_holders"]
            self.final_index = holder_dict["final_index"]
            holder_list = holder_dict["holder_list"]
            self.node_degree = holder_dict["node_degree"] if "node_degree" in holder_dict else None

            self.etype_len = self.get_etype_len()
            self.holder_list.extend(holder_list)
            self.gnn_model = AutoHeterGNN(
                hidden_size=self.hidden_size,
                num_layers=len(self.config.samples),
                layer_type=self.config.sage_layer_type,
                etype_len=self.etype_len,
                act=self.config.sage_act,
                alpha_residual=self.config.sage_alpha,
                interact_mode=self.config.sage_layer_type,
                use_degree_norm=use_degree_norm,
                return_weight=self.return_weight)

        self.total_gpups_slots = [int(self.config.nodeid_slot)] + \
                [int(i) for i in self.config.slots]
        self.real_emb_size_list = [self.emb_size] * len(self.total_gpups_slots)

        self.loss = None

        predictions = self.forward()
        loss, v_loss = self.loss_func(predictions)
        self.loss = loss

        # for visualization
        model_util.paddle_print(v_loss)
        self.visualize_loss, self.batch_count = model_util.loss_visualize(
            v_loss)

        if self.is_predict:
            if self.config.sage_mode:
                node_index = paddle.gather(self.nodeid_slot_holder,
                                           self.final_index)
            else:
                node_index = self.nodeid_slot_holder
            model_util.dump_embedding(config, predictions["src_nfeat"],
                                      node_index)

    def get_etype_len(self):
        """ get length of etype list """
        etype2files = helper.parse_files(self.config.etype2files)
        etype_list = util.get_all_edge_type(etype2files, self.config.symmetry)
        log.info("len of etypes: %s" % len(etype_list))
        return len(etype_list)

    def forward(self):
        """ forward """
        hcl_logits_list = None

        id_embedding, slot_embedding_list = model_util.get_sparse_embedding(
            self.config, self.nodeid_slot_holder, self.discrete_slot_holders,
            self.discrete_slot_lod_holders, self.show_clk, self._use_cvm,
            self.emb_size)

        # merge id_embedding and slot_embedding_list here
        feature = paddle.add_n([id_embedding] + slot_embedding_list)
        empty_flag = paddle.cast(paddle.mean(feature ** 2, -1, keepdim=True) > 1e-3, dtype="float32")
    
        if self.config.softsign:
            log.info("using softsign in feature_mode (sum)")
            feature = paddle.nn.functional.softsign(feature)

        if self.config.sage_mode:

            if not self.is_predict:
                graph_holders = model_util.remove_leakage_edges_gnn(self.graph_holders,
                                                                    self.etype_len,
                                                                    self.final_index,
                                                                    empty_flag,
                                                                    self.return_weight)
            else:
                graph_holders = self.graph_holders


            feature = self.gnn_model(graph_holders, feature, self.node_degree)

            if (not self.is_predict) and self.config.hcl:
                hcl_logits_list = model_util.hcl(self.config, self.gnn_model.hcl_buffer,
                                                 self.graph_holders, empty_flag)
            # remove empty feature. Don't train empty nodes
            feature = feature * empty_flag[:paddle.shape(feature)[0]]

            feature = paddle.gather(feature, self.final_index)

        feature = paddle.reshape(feature, shape=[-1, 2, self.emb_size])

        src_feat = feature[:, 0:1, :]
        dsts_feat_all = [feature[:, 1:, :]]
        for neg in range(self.neg_num):
            dsts_feat_all.append(
                F.layers.shuffle_batch(dsts_feat_all[0]))
        dsts_feat = paddle.concat(dsts_feat_all, axis=1)

        logits = paddle.matmul(
            src_feat, dsts_feat,
            transpose_y=True)  # [batch_size, 1, neg_num+1]
        logits = paddle.squeeze(logits, axis=[1])

        predictions = {}
        predictions["logits"] = logits  # [B, neg_num + 1]
        predictions["nfeat"] = feature  # [B, 2, d]
        predictions["src_nfeat"] = src_feat  # [B, 1, d]
        if hcl_logits_list is not None:
            predictions["hcl_logits"] = hcl_logits_list

        return predictions

    def loss_func(self, predictions, label=None):
        """loss_func"""
        if "loss" not in self.config.loss_type:
            loss_type = "%s_loss" % self.config.loss_type
        else:
            loss_type = self.config.loss_type

        loss_count = 1
        loss = getattr(Loss, loss_type)(self.config, predictions)

        if self.config.gcl_loss:
            gcl_loss = getattr(Loss, self.config.gcl_loss)(self.config,
                                                           predictions)
            loss += gcl_loss
            loss_count += 1

        hcl_logits_list = predictions.get("hcl_logits")
        if (not self.is_predict) and (hcl_logits_list is not None):
            hcl_loss = Loss.hcl_loss(self.config, hcl_logits_list)
            loss += hcl_loss
            loss_count += 1

        # for visualization
        v_loss = loss / self.config.batch_size / loss_count

        return loss, v_loss
