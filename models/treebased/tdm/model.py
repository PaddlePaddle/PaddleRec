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

import numpy as np
import paddle
import paddle.fluid as fluid
import math


def _layer_dot(inputs, node):
    """
    dot product, e.g: [2, 1, 128] * ( expand([1, 128, 1])->[2, 128, 1] )
    """
    input_re = paddle.unsqueeze(inputs, axis=[2])
    dot_res = paddle.matmul(node, input_re)
    return dot_res


def _layer_sub(inputs, node):
    """
    layer_sub, input(-1, emb_size), node(-1, n, emb_size)
    """
    input_re = paddle.unsqueeze(inputs, axis=[1])
    sub_res = paddle.subtract(input_re, node)
    return sub_res


def _layer_mul(inputs, node):
    """
    layer_mul, input(-1, emb_size), node(-1, n, emb_size)
    """
    input_re = paddle.unsqueeze(inputs, axis=[1])
    mul_res = paddle.multiply(input_re, node)
    return mul_res


def middle_transform(output_dim, x, y):
    dot_res = _layer_dot(x, y)
    sub_res = _layer_sub(x, y)
    mul_res = _layer_mul(x, y)

    #paddle.static.Print(y, summarize=-1)
    hiddens_ = paddle.concat(x=[mul_res, sub_res, dot_res], axis=-1)
    '''    
    for idx in range(len(layer_list)):
        hiddens_ = paddle.static.nn.fc(
                x = hiddens_,
                size = layer_list[idx],
                activation = act_list[idx],
                weight_attr=paddle.ParamAttr(name="relu.w" + str(idx)),
                bias_attr=fluid.ParamAttr(name="relu.b" + str(idx)))
    '''
    #paddle.static.Print(hiddens_, summarize=-1)
    hiddens_ = paddle.static.nn.fc(
        x=hiddens_,
        size=output_dim,
        num_flatten_dims=2,
        activation='relu',
        weight_attr=paddle.ParamAttr(
            name="relu.w",
            initializer=paddle.fluid.initializer.NormalInitializer(seed=1)),
        bias_attr=fluid.ParamAttr(
            name="relu.b",
            initializer=paddle.fluid.initializer.ConstantInitializer(
                value=0.1)))
    #hiddens_ = paddle.nn.functional.dropout(hiddens_, 0.1)

    #paddle.static.Print(hiddens_, summarize=-1)
    #hiddens_ = paddle.fluid.layers.dropout(hiddens_, 0.1, seed=1)
    hiddens_ = paddle.static.nn.fc(
        x=hiddens_,
        size=2,
        activation=None,
        weight_attr=paddle.fluid.ParamAttr(
            name="cos_sim.w",
            initializer=paddle.fluid.initializer.NormalInitializer(seed=1)),
        bias_attr=fluid.ParamAttr(
            name="cos_sim.b",
            initializer=paddle.fluid.initializer.ConstantInitializer(
                value=0.1)))
    return hiddens_


def dnn_model_define(output_dim,
                     user_input,
                     unit_id_emb,
                     node_emb_size=24,
                     fea_groups="20,20,10,10,2,2,2,1,1,1",
                     active_op='prelu',
                     use_batch_norm=True,
                     with_att=False):
    print("TDM DNN")
    '''
    user_input = paddle.concat(
        user_input, axis=1)  # [bs, total_group_length, emb_size]
    '''
    dout = middle_transform(output_dim, user_input, unit_id_emb)
    return dout
