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


class ShareBottomLayer(nn.Layer):
    def __init__(self, feature_size, task_num, bottom_size, tower_size):
        super(ShareBottomLayer, self).__init__()

        self.task_num = task_num

        self._param_bottom = self.add_sublayer(
            name='bottom',
            sublayer=nn.Linear(
                feature_size,
                bottom_size,
                #weight_attr=nn.initializer.Constant(value=0.1),
                bias_attr=paddle.ParamAttr(learning_rate=1.0),
                #bias_attr=nn.initializer.Constant(value=0.1),
                name='bottom'))

        self._param_tower = []
        self._param_tower_out = []
        for i in range(0, self.task_num):
            linear = self.add_sublayer(
                name='tower_' + str(i),
                sublayer=nn.Linear(
                    bottom_size,
                    tower_size,
                    weight_attr=nn.initializer.Constant(value=0.1),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    name='tower_' + str(i)))
            self._param_tower.append(linear)

            linear = self.add_sublayer(
                name='tower_out_' + str(i),
                sublayer=nn.Linear(
                    tower_size,
                    2,
                    weight_attr=nn.initializer.Constant(value=0.1),
                    bias_attr=nn.initializer.Constant(value=0.1),
                    name='tower_out_' + str(i)))
            self._param_tower_out.append(linear)

    def forward(self, input_data):
        bottom_tmp = self._param_bottom(input_data)
        bottom_out = F.relu(bottom_tmp)

        output_layers = []
        for i in range(0, self.task_num):
            cur_tower = self._param_tower[i](bottom_out)
            cur_tower = F.relu(cur_tower)
            out_tmp = self._param_tower_out[i](cur_tower)
            out = F.softmax(out_tmp)
            out_clip = paddle.clip(out, min=1e-15, max=1.0 - 1e-15)
            output_layers.append(out_clip)

        return output_layers
