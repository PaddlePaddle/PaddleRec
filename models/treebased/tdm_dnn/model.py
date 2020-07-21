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
import paddle
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.model import ModelBase
import math


class Model(ModelBase):
    def __init__(self, config):
        ModelBase.__init__(self, config)

    def _init_hyper_parameters(self):
        # tree meta hyper parameters
        self.hidden_layers = envs.get_global_env(
            "hyper_parameters.hidden_layers", [128, 64, 24])

        self.max_layers = envs.get_global_env("hyper_parameters.max_layers", 4)
        self.node_nums = envs.get_global_env("hyper_parameters.node_nums", 26)
        self.leaf_node_nums = envs.get_global_env(
            "hyper_parameters.leaf_node_nums", 13)
        self.output_positive = envs.get_global_env(
            "hyper_parameters.output_positive", True)
        self.layer_node_num_list = envs.get_global_env(
            "hyper_parameters.layer_node_num_list", [2, 4, 7, 12])
        self.child_nums = envs.get_global_env("hyper_parameters.child_nums", 2)
        self.tree_layer_path = envs.get_global_env(
            "hyper_parameters.tree.tree_layer_path", None)

        # model training hyper parameter
        self.node_emb_size = envs.get_global_env(
            "hyper_parameters.node_emb_size", 64)
        self.neg_sampling_list = envs.get_global_env(
            "hyper_parameters.neg_sampling_list", [1, 2, 3, 4])
        self.fea_group = envs.get_global_env(
            "hyper_parameters.fea_group", [20, 20, 10, 10, 2, 2, 2, 1, 1, 1])
        self.item_nums = envs.get_global_env("hyper_parameters.item_nums", 69)
        # model infer hyper parameter
        self.topK = envs.get_global_env(
            "hyper_parameters.topK",
            1, )
        self.batch_size = envs.get_global_env(
            "dataset.dataset_train.batch_size", 100)

    def net(self, input, is_infer=False):
        if not is_infer:
            return self.train_net(input)
        else:
            return self.infer_net(input)

    def train_net(self, input):
        self.tdm_dnn_net(input)
        self.create_info()
        self.avg_loss()
        self.metrics()

    def infer_net(self, input):
        self.create_first_layer()
        self.tdm_infer_net(input)

    def input_data(self, is_infer=False, **kwargs):
        user_input = [
            fluid.data(
                name="item_" + str(i + 1), shape=[None, 1], dtype="int64")
            for i in range(self.item_nums)
        ]

        user_input_mask = [
            fluid.data(
                name="item_mask_" + str(i + 1),
                shape=[None, 1],
                dtype="float32") for i in range(self.item_nums)
        ]

        item_label = fluid.data(
            name="item_label",
            shape=[None, 1],
            dtype="int64", )

        return user_input + user_input_mask + [item_label]

    """ -------- Train network detail ------- """

    def train_input(self):
        user_input = [
            fluid.data(
                name="item_" + str(i + 1), shape=[None, 1], dtype="int64")
            for i in range(self.item_nums)
        ]

        user_input_mask = [
            fluid.data(
                name="item_mask_" + str(i + 1),
                shape=[None, 1],
                dtype="float32") for i in range(self.item_nums)
        ]

        item_label = fluid.data(
            name="item_label",
            shape=[None, 1],
            dtype="int64", )

        return user_input + user_input_mask + [item_label]

    def tdm_dnn_net(self, input):
        """
        tdm训练网络的主要流程部分
        """
        print("in train net")
        is_distributed = True if envs.get_trainer() == "CtrTrainer" else False

        user_feature = input[0:self.item_nums]
        user_feature_mask = input[self.item_nums:-1]
        item_label = input[-1]

        # 根据输入的item的正样本在给定的树上进行负采样
        # sample_nodes 是采样的node_id的结果，包含正负样本
        # sample_label 是采样的node_id对应的正负标签
        # sample_mask 是为了保持tensor维度一致，padding部分的标签，若为0，则是padding的虚拟node_id

        if self.check_version():
            with fluid.device_guard("cpu"):
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
                    dtype='int64')
        else:
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
                dtype='int64')

        # 查表得到每个节点的Embedding
        sample_nodes_emb = [
            fluid.layers.embedding(
                input=sample_nodes[i],
                is_sparse=True,
                size=[self.node_nums, self.node_emb_size],
                param_attr=fluid.ParamAttr(name="TDM_Tree_Emb"))
            for i in range(self.max_layers)
        ]

        # 此处进行Reshape是为了之后层次化的分类器训练
        sample_nodes_emb = [
            fluid.layers.reshape(sample_nodes_emb[i], [
                -1, self.neg_sampling_list[i] + self.output_positive,
                self.node_emb_size
            ]) for i in range(self.max_layers)
        ]

        user_feature_emb = self.get_user_emb(user_feature)

        # TDM原始论文 各层共享一个分类器, 因此将同一个batch的所有采样节点拼到一起 [batch_size, node_nums, node_emb_dim]
        sample_nodes_emb = fluid.layers.concat(sample_nodes_emb, axis=1)

        # 过交互结构，得到若干fea_group(time_window)的特征与sample_node组合成的输入
        user_node_concat = self.dnn_interactive_layer(user_feature_emb, sample_nodes_emb,
                                                      user_feature_mask)

        # 过3层分类器
        fcs = [user_node_concat]
        for index, size in enumerate(self.hidden_layers):
            output = self.dnn_layer(fcs[-1], size, fcs[-1].shape[2],
                                    str(index), True)
            fcs.append(output)

        # 计算最后的prob
        tdm_fc = fluid.layers.fc(
            input=fcs[-1],
            size=2,
            act=None,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="tdm.cls_fc.weight",
                initializer=fluid.initializer.Normal(
                    scale=1.0 / math.sqrt(fcs[-1].shape[2]))),
            bias_attr=fluid.ParamAttr(
                name="tdm.cls_fc.bias",
                initializer=fluid.initializer.Constant(0.1)))

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

    def get_user_emb(self, user_feature):
        # user feature查表并做seqpool sum

        def embedding_layer(input):
            emb = fluid.layers.embedding(
                input=input,
                is_sparse=True,
                size=[self.node_nums, self.node_emb_size],
                param_attr=fluid.ParamAttr(name="TDM_Tree_Emb"))
            return emb

        user_feature_emb = list(map(embedding_layer, user_feature))

        return user_feature_emb

    def dnn_interactive_layer(self, user_feature_emb, sample_nodes_emb, user_feature_mask):
        # user_feature_emb: list[ [batch_size, emb] ..{item_nums}.. [batch_size, emb]]
        # sample_nodes_emb: [batch_size, node_nums, emb]

        user_feature_emb_unsqueeze = [
            fluid.layers.unsqueeze(
                user_feature_emb[i], axes=1) for i in range(self.item_nums)
        ]
        # user_feature_emb: list[ [batch_size, 1, emb] ..{item_nums}.. [batch_size, 1, emb]]

        user_feature_emb_expand = [
            fluid.layers.expand(
                user_feature_emb_unsqueeze[i],
                expand_times=[1, sample_nodes_emb.shape[1], 1])
            for i in range(self.item_nums)
        ]
        # user_feature_emb: list[ [batch_size, node_nums, emb] ..{item_nums}.. [batch_size, node_nums, emb]]

        # user_input_mask: list [ [batch_size, 1] ..{item_nums}.. [batch_size, 1]
        user_feature_mask_unsqueeze = [
            fluid.layers.unsqueeze(
                user_feature_mask[i], axes=1) for i in range(self.item_nums)
        ]

        user_feature_mask_expand = [
            fluid.layers.expand(
                user_feature_mask_unsqueeze[i],
                expand_times=[1, sample_nodes_emb.shape[1], 1])
            for i in range(self.item_nums)
        ]
        # user_feature_mask_expand: list[ [batch_size, node_nums, 1] ..{item_nums}.. [batch_size, node_nums, 1]]

        # att_res: list[ [batch_size, node_nums, emb] ..{item_nums}.. [batch_size, node_nums, emb] ]
        user_feature = [
            fluid.layers.elementwise_mul(user_feature_emb_expand[i],
                                         user_feature_mask_expand[i])
            for i in range(self.item_nums)
        ]

        # fea_group: [20, 20, 10, 10, 2, 2, 2, 1, 1, 1], do average in each fea_group
        start_item_index = 0
        end_item_index = 0
        fea_group_output = []
        for feasign_grout_length in self.fea_group:
            end_item_index = start_item_index + feasign_grout_length
            current_fea_group = user_feature[start_item_index:end_item_index]
            current_fea_group_sum = fluid.layers.sum(current_fea_group)

            current_fea_group_average = fluid.layers.scale(
                current_fea_group_sum, 1.0 / float(feasign_grout_length))
            # current_fea_group_average: [batch_size, node_nums, emb]
            fea_group_output.append(current_fea_group_average)
            start_item_index = end_item_index

        fea_grouo_concat = fluid.layers.concat(
            fea_group_output + [sample_nodes_emb], axis=2)
        # fea_grouo_concat: [batch_size, node_nums, emb * len(fea_group) + emb]

        return fea_grouo_concat

    def dnn_layer(self, input, size=128, pre_shape=128, name="",
                  use_act=False):
        # 分类器网络，DNN接prelu及batchnorm

        fc = fluid.layers.fc(
            input=input,
            size=size,
            num_flatten_dims=2,
            act="relu",
            param_attr=fluid.ParamAttr(
                name="cls.fc_{}.weight".format(name),
                initializer=fluid.initializer.Normal(scale=1.0 /
                                                     math.sqrt(pre_shape))),
            bias_attr=fluid.ParamAttr(
                name="cls.fc_{}.bias".format(name),
                initializer=fluid.initializer.Constant(0.1)), )

        return fc

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
                                             num_thresholds=2**12,
                                             slide_steps=20)
        self._metrics["AUC"] = auc
        self._metrics["BATCH_AUC"] = batch_auc
        self._metrics["BATCH_LOSS"] = self._cost

    """ -------- Infer network detail ------- """

    def infer_input(self):
        user_input = [
            fluid.data(
                name="fea_" + str(i), shape=[1], lod_level=1, dtype="int64")
            for i in range(self.item_nums)
        ]

        item_label = fluid.data(
            name="item_label",
            shape=[None, 1],
            dtype="int64", )

        return user_input + [item_label]

    def get_layer_list(self):
        """get layer list from layer_list.txt"""
        layer_list = []
        with open(self.tree_layer_path, 'r') as fin:
            for line in fin.readlines():
                l = []
                layer = (line.split('\n'))[0].split(',')
                for node in layer:
                    if node:
                        l.append(node)
                layer_list.append(l)
        self.layer_list = layer_list

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
        print("self.batch_size {}".format(self.batch_size))
        for id in first_layer_node:
            node_list.append(
                fluid.layers.fill_constant(
                    [self.batch_size, 1], value=int(id), dtype='int64'))
            mask_list.append(
                fluid.layers.fill_constant(
                    [self.batch_size, 1], value=0, dtype='int64'))
        self.first_layer_node = fluid.layers.concat(node_list, axis=1)
        fluid.layers.Print(self.first_layer_node, message="first_layer_node")
        self.first_layer_node_mask = fluid.layers.concat(mask_list, axis=1)

    def tdm_infer_net(self, input):
        """
        infer的主要流程
        infer的基本逻辑是：从上层开始（具体层idx由树结构及TopK值决定）
        1、依次通过每一层分类器，得到当前层输入的指定节点的prob
        2、根据prob值大小，取topK的节点，取这些节点的孩子节点作为下一层的输入
        3、循环1、2步骤，遍历完所有层，得到每一层筛选结果的集合
        4、将筛选结果集合中的叶子节点，拿出来再做一次topK，得到最终的召回输出
        """
        print("in infer net")
        user_feature = input[0:-1]
        item_label = input[-1]
        node_score = []
        node_list = []

        current_layer_node = self.first_layer_node
        current_layer_node_mask = self.first_layer_node_mask

        user_feature_emb = self.get_user_emb(user_feature)

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
                size=[self.node_nums, self.node_emb_size],
                param_attr=fluid.ParamAttr(name="TDM_Tree_Emb"))

            user_feature_emb = self.get_user_emb(user_feature)

            # TDM原始论文 各层共享一个分类器, 因此将同一个batch的所有采样节点拼到一起 [batch_size, node_nums, node_emb_dim]
            sample_nodes_emb = fluid.layers.reshape(
                node_emb, [-1, current_layer_node_num, self.node_emb_size])

            # expand一下uesr feature的维度，与node数量相等 [batch_size, node_nums, feature_emb_dim * feature_nums]
            user_feature_emb = fluid.layers.expand(
                user_feature_emb,
                expand_times=[1, sample_nodes_emb.shape[1], 1])

            # cocat user与node的emb
            user_node_emb = fluid.layers.concat(
                [user_feature_emb, sample_nodes_emb], axis=2)

            # 过3层分类器
            fcs = [user_node_emb]
            print("self.hidden_layers {}".format(self.hidden_layers))
            for index, size in enumerate(self.hidden_layers):
                print("index: {}".format(index))
                output = self.dnn_layer(fcs[-1], size, fcs[-1].shape[2],
                                        str(index), True)
                fcs.append(output)

            # 过最终的判别分类器
            tdm_fc = fluid.layers.fc(
                input=fcs[-1],
                size=2,
                act=None,
                num_flatten_dims=2,
                param_attr=fluid.ParamAttr(name="tdm.cls_fc.weight"),
                bias_attr=fluid.ParamAttr(name="tdm.cls_fc.bias"))

            prob = fluid.layers.softmax(tdm_fc)
            positive_prob = fluid.layers.slice(
                prob, axes=[2], starts=[1], ends=[2])
            prob_re = fluid.layers.reshape(positive_prob,
                                           [-1, current_layer_node_num])

            # 过滤掉padding产生的无效节点（node_id=0）
            node_zero_mask = fluid.layers.cast(current_layer_node, 'bool')
            node_zero_mask = fluid.layers.cast(node_zero_mask, 'float32')
            prob_re = prob_re * node_zero_mask

            # 在当前层的分类结果中取topK，并将对应的score及node_id保存下来
            k = self.topK
            if current_layer_node_num < self.topK:
                k = current_layer_node_num
            _, topk_i = fluid.layers.topk(prob_re, k)

            # index_sample op根据下标索引tensor对应位置的值
            # 若paddle版本 1.8.x, 调用方式为paddle.layers.index_sample
            # 若paddle版本 >= 2.0, 调用方式为paddle.index_sample
            top_node = fluid.contrib.layers.index_sample(current_layer_node,
                                                         topk_i)
            prob_re_mask = prob_re * current_layer_node_mask  # 过滤掉非叶子节点
            topk_value = fluid.contrib.layers.index_sample(prob_re_mask,
                                                           topk_i)
            node_score.append(topk_value)
            node_list.append(top_node)

            # 取当前层topK结果的孩子节点，作为下一层的输入
            if layer_idx < self.max_layers - 1:
                # tdm_child op 根据输入返回其 child 及 child_mask
                # 若child是叶子节点，则child_mask=1，否则为0
                current_layer_node, current_layer_node_mask = fluid.contrib.layers.tdm_child(
                    x=top_node,
                    node_nums=self.node_nums,
                    child_nums=self.child_nums,
                    param_attr=fluid.ParamAttr(name="TDM_Tree_Info"),
                    dtype='int64')

        total_node_score = fluid.layers.concat(node_score, axis=1)
        total_node = fluid.layers.concat(node_list, axis=1)

        # 考虑到树可能是不平衡的，计算所有层的叶子节点的topK
        res_score, res_i = fluid.layers.topk(total_node_score, self.topK)
        res_layer_node = fluid.contrib.layers.index_sample(total_node, res_i)
        res_node = fluid.layers.reshape(res_layer_node, [-1, self.topK, 1])

        # 利用Tree_info信息，将node_id转换为item_id
        tree_info = fluid.default_main_program().global_block().var(
            "TDM_Tree_Info")
        res_node_emb = fluid.layers.gather_nd(tree_info, res_node)

        res_item = fluid.layers.slice(
            res_node_emb, axes=[2], starts=[0], ends=[1])
        self.res_item_re = fluid.layers.reshape(res_item, [-1, self.topK])
        self._infer_results["item"] = self.res_item_re

    def check_version(self):
        """
        Log error and exit when the installed version of paddlepaddle is
        not satisfied.
        """
        err = "TDM-GPU need Paddle version 1.8 or higher is required, " \
            "or a suitable develop version is satisfied as well. \n" \
            "Please make sure the version is good with your code." \

        try:
            fluid.require_version('1.8.0')
            return True
        except Exception as e:
            print(err)
            return False
