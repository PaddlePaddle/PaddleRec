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
import itertools


class FAT_DeepFFMLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes):
        super(FAT_DeepFFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes = layer_sizes
        self.num_fields = sparse_num_field + dense_feature_dim
<<<<<<< HEAD
        self.cen = CENLayer(sparse_feature_number,dense_feature_dim,self.num_fields,sparse_feature_dim, reduction=1)
        self.dnn = DNNLayer(
                sparse_feature_number, 
                sparse_feature_dim,
                dense_feature_dim,
                sparse_num_field, 
                layer_sizes
        )
        self.deepffm = DeepFFM(sparse_feature_number, sparse_feature_dim,dense_feature_dim, sparse_num_field)
=======
        self.cen = CENLayer(
            sparse_feature_number,
            dense_feature_dim,
            self.num_fields,
            sparse_feature_dim,
            reduction=1)
        self.dnn = DNNLayer(sparse_feature_number, sparse_feature_dim,
                            dense_feature_dim, sparse_num_field, layer_sizes)
        self.deepffm = DeepFFM(sparse_feature_number, sparse_feature_dim,
                               dense_feature_dim, sparse_num_field)
>>>>>>> upstream/master
        self.bias = paddle.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, sparse_inputs, dense_inputs):
        # CENLayer 
        cen_out = self.cen(sparse_inputs, dense_inputs)
        # DeepFFMLayer
        y_first_order, dnn_input = self.deepffm(cen_out)
        # DNNLayer
        y_dnn = self.dnn(dnn_input)
        # PredictionLayer
        predict = F.sigmoid(y_first_order + y_dnn + self.bias)

        return predict


<<<<<<< HEAD

class CENLayer(nn.Layer):
    def __init__(self, sparse_feature_number,dense_feature_dim,num_fields,sparse_feature_dim, reduction=8, activation=nn.ReLU()):
=======
class CENLayer(nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 dense_feature_dim,
                 num_fields,
                 sparse_feature_dim,
                 reduction=8,
                 activation=nn.ReLU()):
>>>>>>> upstream/master
        super(CENLayer, self).__init__()
        self.feature_dim = sparse_feature_dim
        self.num_fields = num_fields
        self.sparse_feature_number = sparse_feature_number
        self.dense_feature_dim = dense_feature_dim
        self.init_value_ = 0.1

        #  sparse embedding
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.feature_dim * self.num_fields,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.feature_dim)))))

        # dense part coding
        self.dense_w = paddle.create_parameter(
            shape=[
<<<<<<< HEAD
                1, self.dense_feature_dim,
                self.feature_dim * self.num_fields
=======
                1, self.dense_feature_dim, self.feature_dim * self.num_fields
>>>>>>> upstream/master
            ],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0))

<<<<<<< HEAD
        inputs_num_fields =  num_fields*num_fields
        reduced_num_fields = inputs_num_fields // reduction

        self.pooling = nn.layer.AdaptiveMaxPool1D(output_size=1)
        self.fc = nn.Sequential(
                    ('ReductionLinear', paddle.nn.Linear(inputs_num_fields, reduced_num_fields)),
                    ('ReductionActivation',activation),
                    ('AdditionLinear', paddle.nn.Linear(reduced_num_fields, inputs_num_fields)),
                    ('AdditionActivation', activation)
        )

=======
        inputs_num_fields = num_fields * num_fields
        reduced_num_fields = inputs_num_fields // reduction

        self.pooling = nn.layer.AdaptiveMaxPool1D(output_size=1)
        self.fc = nn.Sequential(('ReductionLinear', paddle.nn.Linear(
            inputs_num_fields,
            reduced_num_fields)), ('ReductionActivation', activation), (
                'AdditionLinear', paddle.nn.Linear(reduced_num_fields,
                                                   inputs_num_fields)),
                                ('AdditionActivation', activation))
>>>>>>> upstream/master

    def forward(self, sparse_inputs, dense_inputs):

        # Embedding 
<<<<<<< HEAD
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)  # [batch_size, sparse_feature_number]
        sparse_embeddings = self.embedding(sparse_inputs_concat)  # [batch_size, sparse_feature_number, sparse_feature_dim] 
      
        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)
        
        feat_embeddings = paddle.concat([sparse_embeddings, dense_embeddings], 1)  # [batch_size, dense_feature_number + feature_number, dense_feature_dim]
        feat_embeddings = paddle.reshape(feat_embeddings,[-1,self.num_fields*self.num_fields,self.feature_dim])
        
        # inputs: emb_inputs, shape = (B, N^2, E) if squared else (B, N, E)
        # output: pooled_inputs, shape = (B,  N^2, 1)
        pooled_inputs = self.pooling(feat_embeddings) 
       
        # Flatten pooled_inputs
        # inputs: pooled_inputs, shape = (B, N^2, 1)
        # output: pooled_inputs, shape = (B, N^2)
        pooled_inputs = paddle.flatten(pooled_inputs, start_axis=1, stop_axis=- 1, name=None)
        
=======
        sparse_inputs_concat = paddle.concat(
            sparse_inputs, axis=1)  # [batch_size, sparse_feature_number]
        sparse_embeddings = self.embedding(
            sparse_inputs_concat
        )  # [batch_size, sparse_feature_number, sparse_feature_dim] 

        dense_inputs_re = paddle.unsqueeze(dense_inputs, axis=2)
        dense_embeddings = paddle.multiply(dense_inputs_re, self.dense_w)

        feat_embeddings = paddle.concat(
            [sparse_embeddings, dense_embeddings], 1
        )  # [batch_size, dense_feature_number + feature_number, dense_feature_dim]
        feat_embeddings = paddle.reshape(
            feat_embeddings,
            [-1, self.num_fields * self.num_fields, self.feature_dim])

        # inputs: emb_inputs, shape = (B, N^2, E) if squared else (B, N, E)
        # output: pooled_inputs, shape = (B,  N^2, 1)
        pooled_inputs = self.pooling(feat_embeddings)

        # Flatten pooled_inputs
        # inputs: pooled_inputs, shape = (B, N^2, 1)
        # output: pooled_inputs, shape = (B, N^2)
        pooled_inputs = paddle.flatten(
            pooled_inputs, start_axis=1, stop_axis=-1, name=None)

>>>>>>> upstream/master
        # Calculate attention weight with dense layer forwardly
        # inputs: pooled_inputs, shape = (B,  N^2)
        # output: attn_w, shape = (B,  N^2)
        attn_w = self.fc(pooled_inputs)
<<<<<<< HEAD
        
=======
>>>>>>> upstream/master

        # Unflatten attention weights and apply it to emb_inputs
        # inputs: attn_w, shape = (B,  N^2)
        # inputs: emb_inputs, shape = (B,  N^2, E)
        # output: outputs, shape = (B,  N^2, E)

        attn_w = paddle.tile(attn_w, repeat_times=[self.feature_dim])
        attn_w = paddle.split(attn_w, num_or_sections=self.feature_dim, axis=1)
<<<<<<< HEAD
        attn_w = paddle.stack(attn_w,axis=2)

        # Multiply attentional weights on field embedding tensors
        outputs = paddle.multiply(feat_embeddings, attn_w) # (B,  N^2, E)

        return outputs
        

class DNNLayer(paddle.nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim,sparse_num_field, layer_sizes, dropout_rate=0.5,is_H=True):
=======
        attn_w = paddle.stack(attn_w, axis=2)

        # Multiply attentional weights on field embedding tensors
        outputs = paddle.multiply(feat_embeddings, attn_w)  # (B,  N^2, E)

        return outputs


class DNNLayer(paddle.nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 sparse_feature_dim,
                 dense_feature_dim,
                 sparse_num_field,
                 layer_sizes,
                 dropout_rate=0.5,
                 is_H=True):
>>>>>>> upstream/master
        super(DNNLayer, self).__init__()
        # self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.num_field = dense_feature_dim + sparse_num_field
        self.layer_sizes = layer_sizes
        self.sparse_num_field = sparse_num_field
        self.is_H = is_H

        if self.is_H:
<<<<<<< HEAD
            self.input_size = int(sparse_feature_dim*self.num_field*(self.num_field-1)/2)
        else:
            self.input_size = int(self.num_field*(self.num_field-1)/2)
=======
            self.input_size = int(sparse_feature_dim * self.num_field *
                                  (self.num_field - 1) / 2)
        else:
            self.input_size = int(self.num_field * (self.num_field - 1) / 2)
>>>>>>> upstream/master

        self.drop_out = paddle.nn.Dropout(p=dropout_rate)

        sizes = [self.input_size] + self.layer_sizes + [1]
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(layer_sizes) + 1):
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
<<<<<<< HEAD
        y_dnn = paddle.reshape(feat_embeddings,[feat_embeddings.shape[0], -1])
=======
        y_dnn = paddle.reshape(feat_embeddings, [feat_embeddings.shape[0], -1])
>>>>>>> upstream/master
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
            y_dnn = self.drop_out(y_dnn)
        return y_dnn


<<<<<<< HEAD
        
class DeepFFM(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field,is_H=True):
        super(DeepFFM, self).__init__()
    
        self.num_field = sparse_num_field + dense_feature_dim # sparse_num_field
        self.sparse_feature_dim = sparse_feature_dim
        self.is_H = is_H
        

    def forward(self,feat_embedding):
        # -------------------- first order term  --------------------
        feat_embedding_ = paddle.sum(feat_embedding, 2)
        y_first_order = paddle.sum(feat_embedding_, 1,keepdim=True)

        # -------------------Field-aware second order term  --------------------
        # feat_embedding [bacth_size,num_field*num_field,feature_dim]
        field_aware_feat_embedding = paddle.reshape(feat_embedding,[-1,self.num_field,self.num_field,self.sparse_feature_dim])
=======
class DeepFFM(nn.Layer):
    def __init__(self,
                 sparse_feature_number,
                 sparse_feature_dim,
                 dense_feature_dim,
                 sparse_num_field,
                 is_H=True):
        super(DeepFFM, self).__init__()

        self.num_field = sparse_num_field + dense_feature_dim  # sparse_num_field
        self.sparse_feature_dim = sparse_feature_dim
        self.is_H = is_H

    def forward(self, feat_embedding):
        # -------------------- first order term  --------------------
        feat_embedding_ = paddle.sum(feat_embedding, 2)
        y_first_order = paddle.sum(feat_embedding_, 1, keepdim=True)

        # -------------------Field-aware second order term  --------------------
        # feat_embedding [bacth_size,num_field*num_field,feature_dim]
        field_aware_feat_embedding = paddle.reshape(
            feat_embedding,
            [-1, self.num_field, self.num_field, self.sparse_feature_dim])
>>>>>>> upstream/master

        field_aware_interaction_list = []
        for i in range(self.num_field):
            for j in range(i + 1, self.num_field):
                if self.is_H:
<<<<<<< HEAD
                    field_aware_out = field_aware_feat_embedding[:, i, j, :] * field_aware_feat_embedding[:, j, i, :]
                else:
                    field_aware_out = paddle.sum(field_aware_feat_embedding[:, i, j, :] *
                               field_aware_feat_embedding[:, j, i, :],
                               1,
                               keepdim=True)
=======
                    field_aware_out = field_aware_feat_embedding[:, i,
                                                                 j, :] * field_aware_feat_embedding[:,
                                                                                                    j,
                                                                                                    i, :]
                else:
                    field_aware_out = paddle.sum(
                        field_aware_feat_embedding[:, i, j, :] *
                        field_aware_feat_embedding[:, j, i, :],
                        1,
                        keepdim=True)
>>>>>>> upstream/master

                field_aware_interaction_list.append(field_aware_out)

        # Iner_product shape: [batch_size, num_fields*(num_fields-1)/2]
        # Hadamard product shape: [batch_size, num_fields*(num_fields-1)/2 * embedding_size]
<<<<<<< HEAD
        y_field_aware_out = paddle.concat(field_aware_interaction_list,axis=1)  

        return y_first_order, y_field_aware_out


=======
        y_field_aware_out = paddle.concat(field_aware_interaction_list, axis=1)

        return y_first_order, y_field_aware_out
>>>>>>> upstream/master
