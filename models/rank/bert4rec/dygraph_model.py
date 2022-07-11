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
import math
import net
import numpy as np


class DygraphModel():
    def create_model(self, config):
        self.num_test_batch = int(
            config.get("hyper_parameters.num_test_user") //
            config.get("runner.data_batch_size"))
        self.test_count = 0
        self.results = [0., 0., 0.]
        _emb_size = config.get("hyper_parameters._emb_size")
        _n_layer = config.get("hyper_parameters._n_layer")
        _n_head = config.get("hyper_parameters._n_head")
        _voc_size = config.get("hyper_parameters._voc_size")
        _max_position_seq_len = config.get(
            "hyper_parameters._max_position_seq_len")
        _sent_types = config.get("hyper_parameters._sent_types")
        hidden_act = config.get("hyper_parameters.hidden_act")
        _dropout = config.get("hyper_parameters._dropout")
        _attention_dropout = config.get("hyper_parameters._attention_dropout")
        initializer_range = config.get("hyper_parameters._param_initializer")
        Bert4Rec = net.BertModel(_emb_size, _n_layer, _n_head, _voc_size,
                                 _max_position_seq_len, _sent_types,
                                 hidden_act, _dropout, _attention_dropout,
                                 initializer_range)
        return Bert4Rec

    # define feeds which convert numpy of batch data to paddle.tensor
    def create_feeds(self, batch_data, config):
        batch_size = config.get("runner.data_batch_size")
        max_len = config.get("hyper_parameters._max_position_seq_len")

        src_ids, pos_ids, input_mask, mask_pos, mask_label = batch_data
        src_ids = paddle.to_tensor(src_ids, dtype='int32').squeeze(0)
        pos_ids = paddle.to_tensor(pos_ids, dtype='int32').squeeze(0)
        input_mask = paddle.to_tensor(input_mask, dtype='int32').squeeze(0)
        mask_pos = paddle.to_tensor(mask_pos, dtype='int32').squeeze(0)
        mask_label = paddle.to_tensor(mask_label, dtype='int64').squeeze(0)
        sent_ids = paddle.zeros(shape=[batch_size, max_len], dtype='int32')
        return src_ids, pos_ids, sent_ids, input_mask, mask_pos, mask_label

    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.0001)
        weight_decay = config.get("hyper_parameters.optimizer.weight_decay",
                                  0.01)
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            grad_clip=nn.ClipGradByGlobalNorm(clip_norm=5.0),
            parameters=dy_model.parameters())
        return optimizer

    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    def create_loss(self, prediction, label):
        mask_lm_loss, lm_softmax = nn.functional.softmax_with_cross_entropy(
            logits=prediction, label=label, return_softmax=True)
        mean_mask_lm_loss = paddle.mean(mask_lm_loss)
        return mean_mask_lm_loss

    def train_forward(self, dy_model, metrics_list, batch_data, config):
        src_ids, pos_ids, sent_ids, input_mask, mask_pos, mask_label = self.create_feeds(
            batch_data, config)

        prediction = dy_model.forward(src_ids, pos_ids, sent_ids, input_mask,
                                      mask_pos)
        loss = self.create_loss(prediction, mask_label)

        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        def evaluate_rec_ndcg_mrr_batch(ratings,
                                        results,
                                        top_k=10,
                                        row_target_position=0):
            ratings = np.array(ratings)
            ratings = ratings[~np.any(np.isnan(ratings), -1)]
            num_rows = len(ratings)
            if num_rows == 0:
                return 0, 0, 0
            ranks = np.argsort(
                np.argsort(
                    -np.array(ratings), axis=-1),
                axis=-1)[:, row_target_position] + 1
            results[2] += np.sum(1 / ranks)
            ranks = ranks[ranks <= top_k]
            results[0] += len(ranks)
            results[1] += np.sum(1 / np.log2(ranks + 1))

        src_ids, pos_ids, sent_ids, input_mask, mask_pos, mask_label = self.create_feeds(
            batch_data[:-1], config)
        batch_size = config.get("runner.data_batch_size")
        candiate = batch_data[-1]
        prediction = dy_model.forward(src_ids, pos_ids, sent_ids, input_mask,
                                      mask_pos)
        pred_ratings = []
        self.test_count += 1
        for i in range(batch_size):
            pred_ratings.append(
                paddle.gather(prediction[i], paddle.to_tensor(candiate[0][i]))
                .numpy())
        evaluate_rec_ndcg_mrr_batch(
            pred_ratings, self.results, top_k=10, row_target_position=0)
        if self.test_count == self.num_test_batch:
            num_user = self.num_test_batch * batch_size
            rec, ndcg, mrr = self.results[0] / num_user, self.results[
                1] / num_user, self.results[2] / num_user
            print(
                "HR@10: %.6f, NDCG@10: %.6f, MRR: %.6f" % (rec, ndcg, mrr),
                end='\n')
        return metrics_list, None
