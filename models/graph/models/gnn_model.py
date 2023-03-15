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
import paddle.fluid as F
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

        self.nodeid_slot_holder, self.show_clk, holder_list = \
                model_util.build_node_holder(self.config.nodeid_slot)
        self.holder_list.extend(holder_list)

        self.discrete_slot_holders, self.discrete_slot_lod_holders, holder_list = \
                model_util.build_slot_holder(self.config.slots)
        self.holder_list.extend(holder_list)

        if self.config.sage_mode:
            use_degree_norm = False if not self.config.use_degree_norm else True
            holder_dict = \
                    model_util.build_graph_holder(self.config.samples, use_degree_norm)
            self.graph_holders = holder_dict["graph_holders"]
            self.final_index = holder_dict["final_index"]
            holder_list = holder_dict["holder_list"]
            self.node_degree = holder_dict[
                "node_degree"] if "node_degree" in holder_dict else None

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
                use_degree_norm=use_degree_norm)

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
        if self.config.softsign:
            log.info("using softsign in feature_mode (sum)")
            feature = paddle.nn.functional.softsign(feature)

        if self.config.sage_mode:
            if self.config.hcl:
                hcl_logits_list = model_util.hcl(self.config, feature,
                                                 self.graph_holders)

            feature = self.gnn_model(self.graph_holders, feature,
                                     self.node_degree)
            feature = paddle.gather(feature, self.final_index)

        feature = paddle.reshape(feature, shape=[-1, 2, self.emb_size])

        src_feat = feature[:, 0:1, :]
        dsts_feat_all = [feature[:, 1:, :]]
        for neg in range(self.neg_num):
            dsts_feat_all.append(
                F.contrib.layers.shuffle_batch(dsts_feat_all[0]))
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
        if hcl_logits_list is not None:
            hcl_loss = Loss.hcl_loss(self.config, hcl_logits_list)
            loss += hcl_loss
            loss_count += 1

        # for visualization
        v_loss = loss / self.config.batch_size / loss_count

        return loss, v_loss
