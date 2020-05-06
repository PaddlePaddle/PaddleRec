# -*- coding=utf-8 -*-
"""
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import math

from fleetrec.core.utils import envs
from fleetrec.core.model import Model as ModelBase


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)
        # tree meta hyper parameters
        self.max_layers = envs.get_global_env(
            "tree_parameters.max_layers", 4, self._namespace)
        self.node_nums = envs.get_global_env(
            "tree_parameters.node_nums", 26, self._namespace)
        self.leaf_node_nums = envs.get_global_env(
            "tree_parameters.leaf_node_nums", 13, self._namespace)
        self.output_positive = envs.get_global_env(
            "tree_parameters.output_positive", True, self._namespace)
        self.layer_node_num_list = envs.get_global_env(
            "tree_parameters.layer_node_num_list", [
                2, 4, 7, 12], self._namespace)
        self.child_nums = envs.get_global_env(
            "tree_parameters.node_nums", 2, self._namespace)
        self.tree_layer_init_path = envs.get_global_env(
            "tree_parameters.tree_layer_init_path", None, self._namespace)

        # model training hyper parameter
        self.node_emb_size = envs.get_global_env(
            "hyper_parameters.node_emb_size", 64, self._namespace)
        self.input_emb_size = envs.get_global_env(
            "hyper_parameters.input_emb_size", 768, self._namespace)
        self.act = envs.get_global_env(
            "hyper_parameters.act", "tanh", self._namespace)
        self.neg_sampling_list = envs.get_global_env(
            "hyper_parameters.neg_sampling_list", [
                1, 2, 3, 4], self._namespace)

        # model infer hyper parameter
        self.topK = envs.get_global_env(
            "hyper_parameters.node_nums", 1, self._namespace)
        self.batch_size = envs.get_global_env(
            "batch_size", 32, "train.reader")

    def train_net(self):
        self.train_input()
        self.tdm_net()
        self.create_info()
        self.avg_loss()
        self.metrics()

    def infer_net(self):
        self.infer_input()
        self.create_first_layer()
        self.tdm_infer_net()

    """ -------- Train network detail ------- """

    def train_input(self):
        input_emb = fluid.data(
            name="input_emb",
            shape=[None, self.input_emb_size],
            dtype="float32",
        )
        self._data_var.append(input_emb)

        item_label = fluid.data(
            name="item_label",
            shape=[None, 1],
            dtype="int64",
        )

        self._data_var.append(item_label)

        if self._platform != "LINUX":
            self._data_loader = fluid.io.DataLoader.from_generator(
                feed_list=self._data_var, capacity=64, use_double_buffer=False, iterable=False)

    def tdm_net(self):
        """
        tdm训练网络的主要流程部分
        """
        is_distributed = True if envs.get_trainer() == "CtrTrainer" else False

        input_emb = self._data_var[0]
        item_label = self._data_var[1]

        # 根据输入的item的正样本在给定的树上进行负采样
        # sample_nodes 是采样的node_id的结果，包含正负样本
        # sample_label 是采样的node_id对应的正负标签
        # sample_mask 是为了保持tensor维度一致，padding部分的标签，若为0，则是padding的虚拟node_id
        sample_nodes, sample_label, sample_mask = fluid.contrib.layers.tdm_sampler(
            x=item_label,
            neg_samples_num_list=self.neg_sampling_list,
            layer_node_num_list=self.layer_node_num_list,
            leaf_node_num=self.leaf_node_nums,
            tree_travel_attr=fluid.ParamAttr(name="TDM_Tree_Travel"),
            tree_layer_attr=fluid.ParamAttr(name="TDM_Tree_Layer"),
            output_positive=self.output_positive,
            output_list=True,
            seed=0,
            tree_dtype='int64',
            dtype='int64'
        )

        # 查表得到每个节点的Embedding
        sample_nodes_emb = [
            fluid.embedding(
                input=sample_nodes[i],
                is_sparse=True,
                size=[self.node_nums, self.node_emb_size],
                param_attr=fluid.ParamAttr(
                    name="TDM_Tree_Emb")
            ) for i in range(self.max_layers)
        ]

        # 此处进行Reshape是为了之后层次化的分类器训练
        sample_nodes_emb = [
            fluid.layers.reshape(sample_nodes_emb[i],
                                 [-1, self.neg_sampling_list[i] +
                                     self.output_positive, self.node_emb_size]
                                 ) for i in range(self.max_layers)
        ]

        # 对输入的input_emb进行转换，使其维度与node_emb维度一致
        input_trans_emb = self.input_trans_layer(input_emb)

        # 分类器的主体网络，分别训练不同层次的分类器
        layer_classifier_res = self.classifier_layer(
            input_trans_emb, sample_nodes_emb)

        # 最后的概率判别FC，将所有层次的node分类结果放到一起以相同的标准进行判别
        # 考虑到树极大可能不平衡，有些item不在最后一层，所以需要这样的机制保证每个item都有机会被召回
        tdm_fc = fluid.layers.fc(input=layer_classifier_res,
                                 size=2,
                                 act=None,
                                 num_flatten_dims=2,
                                 param_attr=fluid.ParamAttr(
                                     name="tdm.cls_fc.weight"),
                                 bias_attr=fluid.ParamAttr(name="tdm.cls_fc.bias"))

        # 将loss打平，放到一起计算整体网络的loss
        tdm_fc_re = fluid.layers.reshape(tdm_fc, [-1, 2])

        # 若想对各个层次的loss辅以不同的权重，则在此处无需concat
        # 支持各个层次分别计算loss，再乘相应的权重
        sample_label = fluid.layers.concat(sample_label, axis=1)
        labels_reshape = fluid.layers.reshape(sample_label, [-1, 1])
        labels_reshape.stop_gradient = True

        # 计算整体的loss并得到softmax的输出
        cost, softmax_prob = fluid.layers.softmax_with_cross_entropy(
            logits=tdm_fc_re, label=labels_reshape, return_softmax=True)

        # 通过mask过滤掉虚拟节点的loss
        sample_mask = fluid.layers.concat(sample_mask, axis=1)
        mask_reshape = fluid.layers.reshape(sample_mask, [-1, 1])
        mask_index = fluid.layers.where(mask_reshape != 0)
        mask_index.stop_gradient = True

        self.mask_cost = fluid.layers.gather_nd(cost, mask_index)

        softmax_prob = fluid.layers.unsqueeze(input=softmax_prob, axes=[1])
        self.mask_prob = fluid.layers.gather_nd(softmax_prob, mask_index)
        self.mask_label = fluid.layers.gather_nd(labels_reshape, mask_index)

        self._predict = self.mask_prob

    def create_info(self):
        fluid.default_startup_program().global_block().create_var(
            name="TDM_Tree_Info",
            dtype=fluid.core.VarDesc.VarType.INT32,
            shape=[self.node_nums, 3 + self.child_nums],
            persistable=True,
            initializer=fluid.initializer.ConstantInitializer(0))
        fluid.default_main_program().global_block().create_var(
            name="TDM_Tree_Info",
            dtype=fluid.core.VarDesc.VarType.INT32,
            shape=[self.node_nums, 3 + self.child_nums],
            persistable=True)

    def avg_loss(self):
        avg_cost = fluid.layers.reduce_mean(self.mask_cost)
        self._cost = avg_cost

    def metrics(self):
        auc, batch_auc, _ = fluid.layers.auc(input=self._predict,
                                             label=self.mask_label,
                                             num_thresholds=2 ** 12,
                                             slide_steps=20)
        self._metrics["AUC"] = auc
        self._metrics["BATCH_AUC"] = batch_auc
        self._metrics["BATCH_LOSS"] = self._cost

    def input_trans_layer(self, input_emb):
        """
        输入侧训练组网
        """
        # 将input映射到与node相同的维度
        input_fc_out = fluid.layers.fc(
            input=input_emb,
            size=self.node_emb_size,
            act=None,
            param_attr=fluid.ParamAttr(name="trans.input_fc.weight"),
            bias_attr=fluid.ParamAttr(name="trans.input_fc.bias"),
        )

        # 将input_emb映射到各个不同层次的向量表示空间
        input_layer_fc_out = [
            fluid.layers.fc(
                input=input_fc_out,
                size=self.node_emb_size,
                act=self.act,
                param_attr=fluid.ParamAttr(
                    name="trans.layer_fc.weight." + str(i)),
                bias_attr=fluid.ParamAttr(name="trans.layer_fc.bias."+str(i)),
            ) for i in range(self.max_layers)
        ]

        return input_layer_fc_out

    def _expand_layer(self, input_layer, node, layer_idx):
        # 扩展input的输入，使数量与node一致，
        # 也可以以其他broadcast的操作进行代替
        # 同时兼容了训练组网与预测组网
        input_layer_unsequeeze = fluid.layers.unsqueeze(
            input=input_layer, axes=[1])
        if not isinstance(node, list):
            input_layer_expand = fluid.layers.expand(
                input_layer_unsequeeze, expand_times=[1, node.shape[1], 1])
        else:
            input_layer_expand = fluid.layers.expand(
                input_layer_unsequeeze, expand_times=[1, node[layer_idx].shape[1], 1])
        return input_layer_expand

    def classifier_layer(self, input, node):
        # 扩展input，使维度与node匹配
        input_expand = [
            self._expand_layer(input[i], node, i) for i in range(self.max_layers)
        ]

        # 将input_emb与node_emb concat到一起过分类器FC
        input_node_concat = [
            fluid.layers.concat(
                input=[input_expand[i], node[i]],
                axis=2) for i in range(self.max_layers)
        ]
        hidden_states_fc = [
            fluid.layers.fc(
                input=input_node_concat[i],
                size=self.node_emb_size,
                num_flatten_dims=2,
                act=self.act,
                param_attr=fluid.ParamAttr(
                    name="cls.concat_fc.weight."+str(i)),
                bias_attr=fluid.ParamAttr(name="cls.concat_fc.bias."+str(i))
            ) for i in range(self.max_layers)
        ]

        # 如果将所有层次的node放到一起计算loss，则需要在此处concat
        # 将分类器结果以batch为准绳concat到一起，而不是layer
        # 维度形如[batch_size, total_node_num, node_emb_size]
        hidden_states_concat = fluid.layers.concat(hidden_states_fc, axis=1)
        return hidden_states_concat

    """ -------- Infer network detail ------- """

    def infer_input(self):
        input_emb = fluid.layers.data(
            name="input_emb",
            shape=[self.input_emb_size],
            dtype="float32",
        )
        self._data_var.append(input_emb)

        if self._platform != "LINUX":
            self._data_loader = fluid.io.DataLoader.from_generator(
                feed_list=self._data_var, capacity=64, use_double_buffer=False, iterable=False)

    def get_layer_list(self):
        """get layer list from layer_list.txt"""
        layer_list = []
        with open(self.tree_layer_init_path, 'r') as fin:
            for line in fin.readlines():
                l = []
                layer = (line.split('\n'))[0].split(',')
                for node in layer:
                    if node:
                        l.append(node)
                layer_list.append(l)
        return layer_list

    def create_first_layer(self):
        """decide which layer to start infer"""
        self.get_layer_list()
        first_layer_id = 0
        for idx, layer_node in enumerate(self.layer_node_num_list):
            if layer_node >= self.topK:
                first_layer_id = idx
                break
        first_layer_node = self.layer_list[first_layer_id]
        self.first_layer_idx = first_layer_id
        node_list = []
        mask_list = []
        for id in node_list:
            node_list.append(fluid.layers.fill_constant(
                [self.batch_size, 1], value=id, dtype='int64'))
            mask_list.append(fluid.layers.fill_constant(
                [self.batch_size, 1], value=0, dtype='int64'))

        self.first_layer_node = fluid.layers.concat(node_list, axis=1)
        self.first_layer_node_mask = fluid.layers.concat(mask_list, axis=1)

    def tdm_infer_net(self, inputs):
        """
        infer的主要流程
        infer的基本逻辑是：从上层开始（具体层idx由树结构及TopK值决定）
        1、依次通过每一层分类器，得到当前层输入的指定节点的prob
        2、根据prob值大小，取topK的节点，取这些节点的孩子节点作为下一层的输入
        3、循环1、2步骤，遍历完所有层，得到每一层筛选结果的集合
        4、将筛选结果集合中的叶子节点，拿出来再做一次topK，得到最终的召回输出
        """
        input_emb = self._data_var[0]

        node_score = []
        node_list = []

        current_layer_node = self.first_layer_node
        current_layer_node_mask = self.first_layer_node_mask
        input_trans_emb = self.input_trans_net.input_fc_infer(input_emb)

        for layer_idx in range(self.first_layer_idx, self.max_layers):
            # 确定当前层的需要计算的节点数
            if layer_idx == self.first_layer_idx:
                current_layer_node_num = self.first_layer_node.shape[1]
            else:
                current_layer_node_num = current_layer_node.shape[1] * \
                    current_layer_node.shape[2]

            current_layer_node = fluid.layers.reshape(
                current_layer_node, [-1, current_layer_node_num])
            current_layer_node_mask = fluid.layers.reshape(
                current_layer_node_mask, [-1, current_layer_node_num])

            node_emb = fluid.embedding(
                input=current_layer_node,
                size=[self.node_nums, self.node_embed_size],
                param_attr=fluid.ParamAttr(name="TDM_Tree_Emb"))

            input_fc_out = self.layer_fc_infer(
                input_trans_emb, layer_idx)

            # 过每一层的分类器
            layer_classifier_res = self.classifier_layer_infer(input_fc_out,
                                                               node_emb,
                                                               layer_idx)

            # 过最终的判别分类器
            tdm_fc = fluid.layers.fc(input=layer_classifier_res,
                                     size=2,
                                     act=None,
                                     num_flatten_dims=2,
                                     param_attr=fluid.ParamAttr(
                                         name="tdm.cls_fc.weight"),
                                     bias_attr=fluid.ParamAttr(name="tdm.cls_fc.bias"))

            prob = fluid.layers.softmax(tdm_fc)
            positive_prob = fluid.layers.slice(
                prob, axes=[2], starts=[1], ends=[2])
            prob_re = fluid.layers.reshape(
                positive_prob, [-1, current_layer_node_num])

            # 过滤掉padding产生的无效节点（node_id=0）
            node_zero_mask = fluid.layers.cast(current_layer_node, 'bool')
            node_zero_mask = fluid.layers.cast(node_zero_mask, 'float')
            prob_re = prob_re * node_zero_mask

            # 在当前层的分类结果中取topK，并将对应的score及node_id保存下来
            k = self.topK
            if current_layer_node_num < self.topK:
                k = current_layer_node_num
            _, topk_i = fluid.layers.topk(prob_re, k)

            # index_sample op根据下标索引tensor对应位置的值
            # 若paddle版本>2.0，调用方式为paddle.index_sample
            top_node = fluid.contrib.layers.index_sample(
                current_layer_node, topk_i)
            prob_re_mask = prob_re * current_layer_node_mask  # 过滤掉非叶子节点
            topk_value = fluid.contrib.layers.index_sample(
                prob_re_mask, topk_i)
            node_score.append(topk_value)
            node_list.append(top_node)

            # 取当前层topK结果的孩子节点，作为下一层的输入
            if layer_idx < self.max_layers - 1:
                # tdm_child op 根据输入返回其 child 及 child_mask
                # 若child是叶子节点，则child_mask=1，否则为0
                current_layer_node, current_layer_node_mask = \
                    fluid.contrib.layers.tdm_child(x=top_node,
                                                   node_nums=self.node_nums,
                                                   child_nums=self.child_nums,
                                                   param_attr=fluid.ParamAttr(
                                                       name="TDM_Tree_Info"),
                                                   dtype='int64')

        total_node_score = fluid.layers.concat(node_score, axis=1)
        total_node = fluid.layers.concat(node_list, axis=1)

        # 考虑到树可能是不平衡的，计算所有层的叶子节点的topK
        res_score, res_i = fluid.layers.topk(total_node_score, self.topK)
        res_layer_node = fluid.contrib.layers.index_sample(total_node, res_i)
        res_node = fluid.layers.reshape(res_layer_node, [-1, self.topK, 1])

        # 利用Tree_info信息，将node_id转换为item_id
        tree_info = fluid.default_main_program().global_block().var("TDM_Tree_Info")
        res_node_emb = fluid.layers.gather_nd(tree_info, res_node)

        res_item = fluid.layers.slice(
            res_node_emb, axes=[2], starts=[0], ends=[1])
        self.res_item_re = fluid.layers.reshape(res_item, [-1, self.topK])

    def input_fc_infer(self, input_emb):
        """
        输入侧预测组网第一部分，将input转换为node同维度
        """
        # 组网与训练时保持一致
        input_fc_out = fluid.layers.fc(
            input=input_emb,
            size=self.node_emb_size,
            act=None,
            param_attr=fluid.ParamAttr(name="trans.input_fc.weight"),
            bias_attr=fluid.ParamAttr(name="trans.input_fc.bias"),
        )
        return input_fc_out

    def layer_fc_infer(self, input_fc_out, layer_idx):
        """
        输入侧预测组网第二部分，将input映射到不同层次的向量空间
        """
        # 组网与训练保持一致，通过layer_idx指定不同层的FC
        input_layer_fc_out = fluid.layers.fc(
            input=input_fc_out,
            size=self.node_emb_size,
            act=self.act,
            param_attr=fluid.ParamAttr(
                name="trans.layer_fc.weight." + str(layer_idx)),
            bias_attr=fluid.ParamAttr(
                name="trans.layer_fc.bias."+str(layer_idx)),
        )
        return input_layer_fc_out

    def classifier_layer_infer(self, input, node, layer_idx):
        # 为infer组网提供的简化版classifier，通过给定layer_idx调用不同层的分类器

        # 同样需要保持input与node的维度匹配
        input_expand = self._expand_layer(input, node, layer_idx)

        # 与训练网络相同的concat逻辑
        input_node_concat = fluid.layers.concat(
            input=[input_expand, node], axis=2)

        # 根据参数名param_attr调用不同的层的FC
        hidden_states_fc = fluid.layers.fc(
            input=input_node_concat,
            size=self.node_emb_size,
            num_flatten_dims=2,
            act=self.act,
            param_attr=fluid.ParamAttr(
                name="cls.concat_fc.weight."+str(layer_idx)),
            bias_attr=fluid.ParamAttr(name="cls.concat_fc.bias."+str(layer_idx)))
        return hidden_states_fc
