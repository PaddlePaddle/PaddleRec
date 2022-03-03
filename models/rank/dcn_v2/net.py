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


class DCN_V2Layer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes, cross_num,
                 is_Stacked, use_low_rank_mixture, low_rank, num_experts):
        super(DCN_V2Layer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.num_field = sparse_num_field + dense_feature_dim
        self.layer_sizes = layer_sizes
        self.cross_num = cross_num
        self.is_Stacked = is_Stacked
        self.use_low_rank_mixture = use_low_rank_mixture
        self.low_rank = low_rank
        self.num_experts = num_experts

        self.init_value_ = 0.1

        # sparse coding
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        self.dense_emb = nn.Linear(self.dense_feature_dim, (
            self.sparse_feature_dim * self.dense_feature_dim))

        self.DeepCrossLayer_ = DeepCrossLayer(
            sparse_num_field, sparse_feature_dim, dense_feature_dim, cross_num,
            use_low_rank_mixture, low_rank, num_experts)

        self.DNN_ = DNNLayer(
            sparse_feature_dim,
            dense_feature_dim,
            sparse_num_field,
            layer_sizes,
            dropout_rate=0.5)

        if self.is_Stacked:
            self.fc = paddle.nn.Linear(
                in_features=self.layer_sizes[-1],
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(self.layer_sizes[-1]))))

        else:
            self.fc = paddle.nn.Linear(
                in_features=self.layer_sizes[-1] +
                (dense_feature_dim + sparse_num_field
                 ) * self.sparse_feature_dim,
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(self.layer_sizes[
                            -1] + dense_feature_dim * sparse_num_field))))

    def forward(self, sparse_inputs, dense_inputs):
        # print("sparse_inputs:",sparse_inputs)
        # print("dense_inputs:",dense_inputs)
        # EmbeddingLayer
        sparse_inputs_concat = paddle.concat(
            sparse_inputs, axis=1)  #Tensor(shape=[bs, 26])
        sparse_embeddings = self.embedding(
            sparse_inputs_concat)  # shape=[bs, 26, dim]

        # print("sparse_embeddings shape:",sparse_embeddings.shape)

        sparse_embeddings_re = paddle.reshape(
            sparse_embeddings,
            shape=[-1, self.sparse_num_field * self.sparse_feature_dim])

        dense_embeddings = self.dense_emb(
            dense_inputs)  # # shape=[bs, 13, dim]

        feat_embeddings = paddle.concat(
            [sparse_embeddings_re, dense_embeddings], 1)
        # print("feat_embeddings:",feat_embeddings.shape)

        # Model Structaul: Stacked or Parallel
        if self.is_Stacked:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(feat_embeddings)
            # MLPLayer
            dnn_output = self.DNN_(cross_out)

            # print('----dnn_output shape----',dnn_output.shape)

            logit = self.fc(dnn_output)
            predict = F.sigmoid(logit)

        else:
            # CrossNetLayer
            cross_out = self.DeepCrossLayer_(feat_embeddings)

            # MLPLayer
            dnn_output = self.DNN_(feat_embeddings)

            last_out = paddle.concat([dnn_output, cross_out], axis=-1)

            # print('----last_out_output shape----',last_out.shape)

            logit = self.fc(last_out)
            predict = F.sigmoid(logit)

        return predict


class DNNLayer(paddle.nn.Layer):
    def __init__(self,
                 sparse_feature_dim,
                 dense_feature_dim,
                 sparse_num_field,
                 layer_sizes,
                 dropout_rate=0.5):
        super(DNNLayer, self).__init__()

        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = dense_feature_dim + sparse_num_field
        self.layer_sizes = layer_sizes
        self.sparse_num_field = sparse_num_field

        self.input_size = int((self.sparse_num_field + self.dense_feature_dim)
                              * self.sparse_feature_dim)

        self.drop_out = paddle.nn.Dropout(p=dropout_rate)

        sizes = [self.input_size] + self.layer_sizes
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes)):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    regularizer=paddle.regularizer.L2Decay(1e-7),
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)

    def forward(self, feat_embeddings):
        # y_dnn = paddle.reshape(feat_embeddings,[feat_embeddings.shape[0], -1])
        y_dnn = feat_embeddings
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
            y_dnn = self.drop_out(y_dnn)
        return y_dnn


class DeepCrossLayer(nn.Layer):
    def __init__(self, sparse_num_field, sparse_feature_dim, dense_feature_dim,
                 cross_num, use_low_rank_mixture, low_rank, num_experts):
        super(DeepCrossLayer, self).__init__()

        self.use_low_rank_mixture = use_low_rank_mixture
        self.input_dim = (
            sparse_num_field + dense_feature_dim) * sparse_feature_dim
        self.num_experts = num_experts
        self.low_rank = low_rank
        self.cross_num = cross_num

        if self.use_low_rank_mixture:
            self.crossNet = CrossNetMix(
                self.input_dim,
                layer_num=self.cross_num,
                low_rank=self.low_rank,
                num_experts=self.num_experts)
        else:
            self.crossNet = CrossNetV2(self.input_dim, self.cross_num)

    def forward(self, feat_embeddings):
        outputs = self.crossNet(feat_embeddings)

        return outputs


class CrossNetV2(nn.Layer):
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()

        self.num_layers = num_layers
        self.cross_layers = nn.LayerList(
            nn.Linear(input_dim, input_dim) for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0  # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i


class CrossNetMix(nn.Layer):
    """ CrossNetMix improves CrossNet by:
        1. add MOE to learn feature interactions in different subspaces
        2. add nonlinear transformations in low-dimensional space
    """

    def __init__(self, in_features, layer_num=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[num_experts, in_features, low_rank],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierNormal())
            for i in range(self.layer_num)
        ])

        # V: (in_features, low_rank)
        self.V_list = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[num_experts, in_features, low_rank],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierNormal())
            for i in range(self.layer_num)
        ])

        # C: (low_rank, low_rank)
        self.C_list = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[num_experts, low_rank, low_rank],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierNormal())
            for i in range(self.layer_num)
        ])

        self.gating = nn.LayerList(
            [nn.Linear(in_features, 1) for i in range(self.num_experts)])

        self.bias = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[in_features, 1],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(value=0.0))
            for i in range(self.layer_num)
        ])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](
                    x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = paddle.matmul(self.V_list[i][expert_id].t(),
                                    x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = paddle.tanh(v_x)
                v_x = paddle.matmul(self.C_list[i][expert_id], v_x)
                v_x = paddle.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = paddle.matmul(self.U_list[i][expert_id],
                                     v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = paddle.stack(
                output_of_experts, axis=2)  # (bs, in_features, num_experts)
            gating_score_of_experts = paddle.stack(
                gating_score_of_experts, axis=1)  # (bs, num_experts, 1)
            moe_out = paddle.matmul(
                output_of_experts, F.softmax(
                    gating_score_of_experts, axis=1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l
