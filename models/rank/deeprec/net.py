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

import paddle.nn as nn
import paddle.nn.functional as F


class DeepRecLayer(nn.Layer):
    def __init__(self, layer_sizes, dp_drop_prob=0.0):
        super(DeepRecLayer, self).__init__()

        self.layer_sizes = layer_sizes
        self.number_of_layers = len(layer_sizes) - 1
        self._dp_drop_prob = dp_drop_prob
        if dp_drop_prob > 0:
            self.drop = nn.Dropout(dp_drop_prob)

        encoder_layer_sizes = layer_sizes
        self._param_encoder = []
        for i in range(self.number_of_layers):
            linear = self.add_sublayer(
                name='encoder_' + str(i),
                sublayer=nn.Linear(
                    encoder_layer_sizes[i],
                    encoder_layer_sizes[i + 1],
                    weight_attr=nn.initializer.XavierUniform(),
                    bias_attr=nn.initializer.Constant(value=0.0),
                    name='encoder_' + str(i)))
            self._param_encoder.append(linear)

        decoder_layer_sizes = list(reversed(layer_sizes))
        self._param_decoder = []
        for i in range(self.number_of_layers):
            linear = self.add_sublayer(
                name='decoder_' + str(i),
                sublayer=nn.Linear(
                    decoder_layer_sizes[i],
                    decoder_layer_sizes[i + 1],
                    weight_attr=nn.initializer.XavierUniform(),
                    bias_attr=nn.initializer.Constant(value=0.0),
                    name='decoder_' + str(i)))
            self._param_decoder.append(linear)

    def forward(self, x):
        for i in range(self.number_of_layers):
            x = self._param_encoder[i](x)
            x = F.selu(x)
        if self._dp_drop_prob > 0:  # apply dropout only on code layer
            x = self.drop(x)

        for i in range(self.number_of_layers):
            x = self._param_decoder[i](x)
            x = F.selu(x)
        return x
