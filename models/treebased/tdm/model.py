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


class FullyConnected3D(object):
    def __init__(self,
                 input_dim,
                 output_dim,
                 active_op='prelu',
                 version="default"):

        self.active_op = active_op
        self.version = version
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, bottom_data):
        print "call FullyConnected3D"

        net_out = paddle.static.nn.fc(
            bottom_data,
            size=self.output_dim,
            num_flatten_dims=2,
            activation=None,
            weight_attr=paddle.framework.ParamAttr(
                name="fc_w_%s" % self.version,
                initializer=fluid.initializer.Normal(scale=1.0 / math.sqrt(
                    (self.input_dim)))),
            bias_attr=fluid.ParamAttr(
                name="fc_b_%s" % self.version,
                initializer=fluid.initializer.Constant(0.1)))

        # net_out = paddle.static.nn.fc(
        #     bottom_data,
        #     size=self.output_dim,
        #     num_flatten_dims=2,
        #     activation=None,
        #     weight_attr=paddle.framework.ParamAttr(
        #         name="fc_w_%s" % self.version,
        #         initializer=fluid.initializer.Constant(1.0 / math.sqrt((self.input_dim)))),
        #     bias_attr=fluid.ParamAttr(
        #         name="fc_b_%s" % self.version,
        #         initializer=fluid.initializer.Constant(0.1)))

        if self.active_op == 'prelu':
            print "in FullyConnected3D use prelu"
            net_out = paddle.static.nn.prelu(
                net_out,
                'channel',
                paddle.framework.ParamAttr(
                    name='alpha_1_%s' % self.version,
                    initializer=paddle.nn.initializer.Constant(0.25)))
        return net_out


class paddle_dnn_layer(object):
    def __init__(self,
                 input_dim,
                 output_dim,
                 active_op='prelu',
                 use_batch_norm=False,
                 version="default"):

        self.active_op = active_op
        self.use_batch_norm = use_batch_norm
        self.version = version
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, bottom_data):
        print "if mx.symbol.FullyConnected"
        out = paddle.static.nn.fc(
            bottom_data,
            size=self.output_dim,
            activation=None,
            weight_attr=paddle.framework.ParamAttr(
                name="fc_w_%s" % self.version,
                initializer=fluid.initializer.Normal(scale=1.0 / math.sqrt(
                    (self.input_dim)))),
            bias_attr=fluid.ParamAttr(
                name="fc_b_%s" % self.version,
                initializer=fluid.initializer.Constant(0.1)))

        # out = paddle.static.nn.fc(
        #     bottom_data,
        #     size=self.output_dim,
        #     activation=None,
        #     weight_attr=paddle.framework.ParamAttr(
        #         name="fc_w_%s" % self.version,
        #         initializer=fluid.initializer.Constant(1.0 / math.sqrt((self.input_dim)))),
        #     bias_attr=fluid.ParamAttr(
        #         name="fc_b_%s" % self.version,
        #         initializer=fluid.initializer.Constant(0.1)))

        if self.use_batch_norm:
            print "if self.use_batch_norm:"
            batch_norm = paddle.nn.BatchNorm(
                self.output_dim,
                epsilon=1e-03,
                param_attr=fluid.ParamAttr(
                    name="bn_gamma_1_%s" % self.version,
                    initializer=fluid.initializer.Constant(1.0)),
                bias_attr=fluid.ParamAttr(
                    name="bn_bias_1_%s" % self.version,
                    initializer=fluid.initializer.Constant(0.0)))
            out = batch_norm(out)

        if self.active_op == 'prelu':
            print "if self.active_op == 'prelu':"
            out = paddle.static.nn.prelu(
                out,
                'channel',
                paddle.framework.ParamAttr(
                    name='alpha_1_%s' % self.version,
                    initializer=paddle.nn.initializer.Constant(0.25)))

        return out


def dnn_model_define(user_input,
                     unit_id_emb,
                     label,
                     node_emb_size=24,
                     fea_groups="20,20,10,10,2,2,2,1,1,1",
                     active_op='prelu',
                     use_batch_norm=True,
                     with_att=False,
                     is_infer=False,
                     topk=10):
    fea_groups = [int(s) for s in fea_groups.split(',')]
    total_group_length = np.sum(np.array(fea_groups))
    print "fea_groups", fea_groups, "total_group_length", total_group_length, "eb_dim", node_emb_size

    layer_data = []
    # start att
    if with_att:
        print("TDM Attention DNN")
        att_user_input = paddle.concat(
            user_input, axis=1)  # [bs, total_group_length, emb_size]
        att_node_input = fluid.layers.expand(
            unit_id_emb, expand_times=[1, total_group_length, 1])
        att_din = paddle.concat(
            [att_user_input, att_user_input * att_node_input, att_node_input],
            axis=2)

        att_active_op = 'prelu'
        att_layer_arr = []
        att_layer1 = FullyConnected3D(
            3 * node_emb_size, 36, active_op=att_active_op, version=1)
        att_layer_arr.append(att_layer1)
        att_layer2 = FullyConnected3D(
            36, 1, active_op=att_active_op, version=2)
        att_layer_arr.append(att_layer2)

        layer_data.append(att_din)
        for layer in att_layer_arr:
            layer_data.append(layer.call(layer_data[-1]))
        att_dout = layer_data[-1]

        att_dout = fluid.layers.expand(
            att_dout, expand_times=[1, 1, node_emb_size])
        user_input = att_user_input * att_dout
    else:
        print("TDM DNN")
        user_input = paddle.concat(
            user_input, axis=1)  # [bs, total_group_length, emb_size]
    # end att

    idx = 0
    grouped_user_input = []
    for group_length in fea_groups:
        block_before_sum = paddle.slice(
            user_input, axes=[1], starts=[idx], ends=[idx + group_length])
        block = paddle.sum(block_before_sum, axis=1) / group_length
        grouped_user_input.append(block)
        idx += group_length
    grouped_user_input = paddle.concat(
        grouped_user_input, axis=1)  # [bs, 10 * emb_size]

    din = paddle.concat(
        [grouped_user_input, paddle.squeeze(
            unit_id_emb, axis=1)], axis=1)

    net_version = "d"
    layer_arr = []
    layer1 = paddle_dnn_layer(
        11 * node_emb_size,
        128,
        active_op=active_op,
        use_batch_norm=use_batch_norm,
        version="%d_%s" % (1, net_version))
    layer_arr.append(layer1)
    layer2 = paddle_dnn_layer(
        128,
        64,
        active_op=active_op,
        use_batch_norm=use_batch_norm,
        version="%d_%s" % (2, net_version))
    layer_arr.append(layer2)
    layer3 = paddle_dnn_layer(
        64,
        24,
        active_op=active_op,
        use_batch_norm=use_batch_norm,
        version="%d_%s" % (3, net_version))
    layer_arr.append(layer3)
    layer4 = paddle_dnn_layer(
        24,
        2,
        active_op='',
        use_batch_norm=False,
        version="%d_%s" % (4, net_version))
    layer_arr.append(layer4)

    layer_data.append(din)
    for layer in layer_arr:
        layer_data.append(layer.call(layer_data[-1]))
    dout = layer_data[-1]

    if is_infer:
        softmax_prob = paddle.nn.functional.softmax(dout)
        positive_prob = paddle.slice(
            softmax_prob, axes=[1], starts=[1], ends=[2])
        prob_re = paddle.reshape(positive_prob, [-1])

        _, topk_i = paddle.topk(prob_re, k=topk)
        topk_node = paddle.index_select(label, topk_i)
        return topk_node
    else:
        cost, softmax_prob = paddle.nn.functional.softmax_with_cross_entropy(
            logits=dout, label=label, return_softmax=True, ignore_index=-1)

        ignore_label = paddle.full_like(label, fill_value=-1)
        avg_cost = paddle.sum(cost) / paddle.sum(
            paddle.cast(
                paddle.not_equal(label, ignore_label), dtype='int32'))

        return avg_cost, softmax_prob
