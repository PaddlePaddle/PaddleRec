# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid


class TDMSampleLayer(nn.layer):
    def __init__(self, config):
        self.neg_sampling_list = config["neg_sampling_list"]
        self.layer_node_num_list = config["layer_node_num_list"]
        self.leaf_node_num = config["leaf_node_num"]
        self.node_nums = config["node_nums"]
        self.return_positiv = config["return_positiv"]
        self.max_layers = config["max_layers"]

        self.emb_lr = config["emb_lr"]
        self.node_emb_size = config["node_emb_size"]

        self.node_embedding = paddle.nn.Embedding(
            self.node_nums,
            self.node_emb_size,
            paddinng_idx=0,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="TDM_Tree_Emb", learning_rate=self.emb_lr)
        )

    def forward(self, item):
        # 根据输入的item的正样本在给定的树上进行负采样
        # sample_nodes 是采样的node_id的结果，包含正负样本
        # sample_label 是采样的node_id对应的正负标签
        # sample_mask 是为了保持tensor维度一致，padding部分的标签，若为0，则是padding的虚拟node_id
        bidword_layers, labels, mask = paddle.fluid.contrib.layers.tdm_sampler(
            x=item,
            neg_samples_num_list=self.neg_sampling_list,
            layer_node_num_list=self.layer_node_num_list,
            leaf_node_num=self.leaf_node_num,
            tree_travel_attr=fluid.ParamAttr(name="TDM_Tree_Travel"),
            tree_layer_attr=fluid.ParamAttr(name="TDM_Tree_Layer"),
            output_positive=self.return_positive,
            output_list=True,
            seed=0,
            dtype='int64')

        # labels = paddle.concat(labels, axis=1)
        # mask = paddle.concat(mask, axis=1)

        bidword_layers = [
            paddle.reshape(bidword_layers[i], [-1, 1]
                           ) for i in range(self.max_layers)
        ]

        # embedding
        pooled_output_y = [
            self.node_embedding(bidword_layers[i]) for i in range(self.max_layers)
        ]
        return pooled_output_y, labels, mask


# class InterActiveLayer(nn.layer):
#     def __init__(self, config):
#         pass

#     def forward(self, user_emb, item_emb):
#          # [0] split_pooler_layer
#         hidden_states_x = self._split_pooler_layer(x)

#         # [1] (mul, sub, dot)
#         sub_res = [
#             self._layer_sub(hidden_states_x[i], y[i])
#             for i in range(self.max_layers)
#         ]

#         mul_res = [
#             self._layer_mul(hidden_states_x[i], y[i])
#             for i in range(self.max_layers)
#         ]

#         dot_res = [
#             self._layer_dot(hidden_states_x[i], y[i])
#             for i in range(self.max_layers)
#         ]

#         # [2] concat
#         concat_out = [
#             fluid.layers.concat(
#                 input=[mul_res[i], sub_res[i], dot_res[i]],
#                 axis=-1,
#             ) for i in range(self.max_layers)
#         ]

#         # [3] fc & tanh
#         hidden_states_fc = [
#             fluid.layers.fc(
#                 input=concat_out[i],
#                 size=self.middle_size,
#                 num_flatten_dims=2,
#                 act="tanh",
#                 param_attr=fluid.ParamAttr(name="mt.inter_fc.weight"),
#                 bias_attr=fluid.ParamAttr(name="mt.inter_fc.bias"),
#             ) for i in range(self.max_layers)
#         ]

#         # [4] reshape & concat
#         hidden_states_fc_re = [
#             fluid.layers.reshape(
#                 hidden_states_fc[i],
#                 [-1, self.layers_samples_list[i] + 1, self.middle_size]
#             ) for i in range(self.max_layers)
#         ]

#         hidden_states_concat = fluid.layers.concat(hidden_states_fc_re, axis=1)

#         # [4] dropout
#         hidden_states = fluid.layers.dropout(x=hidden_states_concat,
#                                              dropout_prob=self.dropout_prob)

#         return hidden_states, hidden_states_x, y

#     def _split_pooler_layer(self, emb_x):
#         """
#                      [0]       [1]             [2]
#         input -->[ dropout --> fc --> layer_fc --> tanh ]--> output
#               768          768    128                    128
#         """
#         # [0] dropout
#         dropout_out = fluid.layers.dropout(
#             x=emb_x,
#             dropout_prob=self.dropout_prob,
#         )

#         # [1] fc
#         fc_out = fluid.layers.fc(
#             input=dropout_out,
#             size=self.middle_size,
#             act=None,
#             param_attr=fluid.ParamAttr(
#                 name="mt.q_fc.weight",
#             ),
#             bias_attr=fluid.ParamAttr(
#                 name="mt.q_fc.bias",
#             ),
#         )

#         # [2] layer_fc & tanh
#         layer_fc_out = [
#             fluid.layers.fc(
#                 input=fc_out,
#                 size=self.middle_size,
#                 act="tanh",
#                 param_attr=fluid.ParamAttr(
#                     name="mt.q_layer_fc.weight." + str(i),
#                 ),
#                 bias_attr=fluid.ParamAttr(
#                     name="mt.q_layer_fc.bias." + str(i),
#                 ),
#             ) for i in range(self.max_layers)
#         ]

#         return layer_fc_out

#     def _layer_dot(self, input, node):
#         """
#         dot product, e.g: [2, 1, 128] * ( expand([1, 128, 1])->[2, 128, 1] )
#         """
#         input_re = fluid.layers.unsqueeze(input, axes=[2])
#         dot_res = fluid.layers.matmul(node, input_re)
#         return dot_res

#     def _layer_sub(self, input, node):
#         """
#         layer_sub, input(-1, emb_size), node(-1, n, emb_size)
#         """
#         input_re = fluid.layers.unsqueeze(input, axes=[1])
#         input_expand = fluid.layers.expand(
#             input_re, expand_times=[1, node.shape[1], 1])
#         sub_res = fluid.layers.elementwise_sub(input_expand, node)
#         return sub_res

#     def _layer_mul(self, input, node):
#         """
#         layer_mul, input(-1, emb_size), node(-1, n, emb_size)
#         """
#         input_re = fluid.layers.unsqueeze(input, axes=[1])
#         input_expand = fluid.layers.expand(
#             input_re, expand_times=[1, node.shape[1], 1])
#         mul_res = fluid.layers.elementwise_mul(input_expand, node)
#         return mul_res


# class LearninngToIndexLayer(nn.layer):
#     def __init__(self, config):
#         self.config = config
#         self.tdm_sample_layer = TDMSampleLayer()
#         self.interactive_layer = InterActiveLayer()

#     def forward(self, emb, item):
#         pooled_output_x = emb
#         bidwords = item

#         item_emb = self.tdm_sample_layer(item)
#         user_emb = [
#             fluid.layers.reshape(
#                 pooled_output_y[i], [-1, self.neg_sampling_list[i] +
#                                     self.return_positive, self.y_index_embed_size]
#             ) for i in range(self.max_layers)
#         ]

#         interacitve_result = self.interactive_layer(user_emb, item_emb)

#         pooled_output_re = fluid.layers.reshape(interacitve_result,
#                                                 [-1, self.y_index_embed_size])
#         logits_fc = fluid.layers.fc(
#             input=pooled_output_re,
#             size=self.num_labels,
#             act=None,
#             param_attr=fluid.ParamAttr(name="tdm.cls_fc.weight"),
#             bias_attr=fluid.ParamAttr(name="tdm.cls_fc.bias"))

#         labels = fluid.layers.reshape(labels, [-1, 1])

#         cost, probs = fluid.layers.softmax_with_cross_entropy(
#             logits=logits_fc,
#             label=labels,
#             return_softmax=True,
#         )

#         mask = fluid.layers.reshape(mask, [-1, 1])

#         mask_index = fluid.layers.where(mask != 0)
#         mask_cost = fluid.layers.gather_nd(cost, mask_index)

#         avg_cost = fluid.layers.reduce_mean(mask_cost)

#         acc = fluid.layers.accuracy(input=probs, label=labels)

#         return avg_cost, acc


# class MiddleTransformLayer(nn.Layer):
#     """
#     MiddleTransform Model
#     """

#     def __init__(self,
#                  dropout_prob,
#                  max_layers,
#                  layers_samples_list,
#                  trace_var=False):
#         """
#         init default params
#         """
#         self.middle_size = 64  # 128
#         self.max_layers = max_layers
#         self.dropout_prob = dropout_prob
#         self.layers_samples_list = layers_samples_list
#         self.trace_var = trace_var

#         self.default_w = 0.1
#         self.default_b = 0.1

#     def create_model(self, x, y):
#         """
#                             [0]                   [1]             [
#                                 2]         [3]          [4]
#         input_x --> [ split_pooler_layer --> (sub, mul, dot) --> concat --> fc + tanh --> dropout ] --> output
#                 768                      128              128,128,1     257           64            64
#                                  input_y -->
#                                          128
#         """


#  def tdm_net(self, input):
#     """
#     tdm训练网络的主要流程部分
#     """
#     is_distributed = True if envs.get_trainer() == "CtrTrainer" else False

#     input_emb = input[0]
#     item_label = input[1]

#     # 根据输入的item的正样本在给定的树上进行负采样
#     # sample_nodes 是采样的node_id的结果，包含正负样本
#     # sample_label 是采样的node_id对应的正负标签
#     # sample_mask 是为了保持tensor维度一致，padding部分的标签，若为0，则是padding的虚拟node_id

#     if self.check_version():
#         with fluid.device_guard("cpu"):
#             sample_nodes, sample_label, sample_mask = fluid.contrib.layers.tdm_sampler(
#                 x=item_label,
#                 neg_samples_num_list=self.neg_sampling_list,
#                 layer_node_num_list=self.layer_node_num_list,
#                 leaf_node_num=self.leaf_node_nums,
#                 tree_travel_attr=fluid.ParamAttr(name="TDM_Tree_Travel"),
#                 tree_layer_attr=fluid.ParamAttr(name="TDM_Tree_Layer"),
#                 output_positive=self.output_positive,
#                 output_list=True,
#                 seed=0,
#                 tree_dtype='int64',
#                 dtype='int64')
#     else:
#         sample_nodes, sample_label, sample_mask = fluid.contrib.layers.tdm_sampler(
#             x=item_label,
#             neg_samples_num_list=self.neg_sampling_list,
#             layer_node_num_list=self.layer_node_num_list,
#             leaf_node_num=self.leaf_node_nums,
#             tree_travel_attr=fluid.ParamAttr(name="TDM_Tree_Travel"),
#             tree_layer_attr=fluid.ParamAttr(name="TDM_Tree_Layer"),
#             output_positive=self.output_positive,
#             output_list=True,
#             seed=0,
#             tree_dtype='int64',
#             dtype='int64')

#     sample_nodes = [
#         fluid.layers.reshape(sample_nodes[i], [-1, 1])
#         for i in range(self.max_layers)
#     ]

#     # 查表得到每个节点的Embedding
#     sample_nodes_emb = [
#         fluid.layers.embedding(
#             input=sample_nodes[i],
#             is_sparse=True,
#             size=[self.node_nums, self.node_emb_size],
#             param_attr=fluid.ParamAttr(name="TDM_Tree_Emb"))
#         for i in range(self.max_layers)
#     ]

#     # 此处进行Reshape是为了之后层次化的分类器训练
#     sample_nodes_emb = [
#         fluid.layers.reshape(sample_nodes_emb[i], [
#             -1, self.neg_sampling_list[i] + self.output_positive,
#             self.node_emb_size
#         ]) for i in range(self.max_layers)
#     ]

#     # 对输入的input_emb进行转换，使其维度与node_emb维度一致
#     input_trans_emb = self.input_trans_layer(input_emb)

#     # 分类器的主体网络，分别训练不同层次的分类器
#     layer_classifier_res = self.classifier_layer(input_trans_emb,
#                                                     sample_nodes_emb)

#     # 最后的概率判别FC，将所有层次的node分类结果放到一起以相同的标准进行判别
#     # 考虑到树极大可能不平衡，有些item不在最后一层，所以需要这样的机制保证每个item都有机会被召回
#     tdm_fc = fluid.layers.fc(
#         input=layer_classifier_res,
#         size=2,
#         act=None,
#         num_flatten_dims=2,
#         param_attr=fluid.ParamAttr(name="tdm.cls_fc.weight"),
#         bias_attr=fluid.ParamAttr(name="tdm.cls_fc.bias"))

#     # 将loss打平，放到一起计算整体网络的loss
#     tdm_fc_re = fluid.layers.reshape(tdm_fc, [-1, 2])

#     # 若想对各个层次的loss辅以不同的权重，则在此处无需concat
#     # 支持各个层次分别计算loss，再乘相应的权重
#     sample_label = fluid.layers.concat(sample_label, axis=1)
#     labels_reshape = fluid.layers.reshape(sample_label, [-1, 1])
#     labels_reshape.stop_gradient = True

#     # 计算整体的loss并得到softmax的输出
#     cost, softmax_prob = fluid.layers.softmax_with_cross_entropy(
#         logits=tdm_fc_re, label=labels_reshape, return_softmax=True)

#     # 通过mask过滤掉虚拟节点的loss
#     sample_mask = fluid.layers.concat(sample_mask, axis=1)
#     mask_reshape = fluid.layers.reshape(sample_mask, [-1, 1])
#     mask_index = fluid.layers.where(mask_reshape != 0)
#     mask_index.stop_gradient = True

#     self.mask_cost = fluid.layers.gather_nd(cost, mask_index)

#     softmax_prob = fluid.layers.unsqueeze(input=softmax_prob, axes=[1])
#     self.mask_prob = fluid.layers.gather_nd(softmax_prob, mask_index)
#     self.mask_label = fluid.layers.gather_nd(labels_reshape, mask_index)

#     self._predict = self.mask_prob
