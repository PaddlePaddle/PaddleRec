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
import os
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model(net, optimizer, model_path, epoch_id, prefix='rec'):
    model_path = os.path.join(model_path, str(epoch_id))
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    paddle.save(net.state_dict(), model_prefix + ".pdparams")
    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    logger.info("Already save model in {}".format(model_path))


def load_model(model_path, net, prefix='rec'):
    logger.info("start load model from {}".format(model_path))
    model_prefix = os.path.join(model_path, prefix)
    param_state_dict = paddle.load(model_prefix + ".pdparams")
    net.set_dict(param_state_dict)


def _mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_static_model(program, model_path, epoch_id, prefix='rec_static'):
    """
    save model to the target path
    """
    model_path = os.path.join(model_path, str(epoch_id))
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    paddle.static.save(program, model_prefix)
    logger.info("Already save model in {}".format(model_path))


def load_static_model(program, model_path, prefix='rec_static'):
    logger.info("start load model from {}".format(model_path))
    model_prefix = os.path.join(model_path, prefix)
    param_state_dict = paddle.static.load(program, model_prefix)
