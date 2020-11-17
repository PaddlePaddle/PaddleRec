# -*- coding=utf-8 -*-
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
from __future__ import print_function

import time
import logging
import os
import numpy as np
import json
import paddle
import paddle.fluid as fluid

from paddlerec.core.utils import envs
from paddlerec.core.trainers.framework.runner import SingleInferRunner
from paddlerec.core.trainers.framework.terminal import TerminalBase
from paddlerec.core.trainer import EngineMode
from paddlerec.core.utils.util import shuffle_files
from paddlerec.core.metric import Metric

from .infer import parse_args, infer_epoch
from .utils import prepare_data

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


class Terminal(TerminalBase):
    def __init__(self, context):
        print("Running CTR-DNN Terminal")
        pass

    def get_example_num(self, file_list):
        count = 0
        for f in file_list:
            last_count = count
            for index, line in enumerate(open(f, 'r')):
                count += 1
            logger.info("file: %s has %s examples" % (f, count - last_count))
        logger.info("Total example: %s" % count)
        return count

    def terminal(self, context):
        work_path = "./benchmark_logs/"
        if not os.path.isdir(work_path):
            os.makedirs(work_path)

        train_files_path = "./train_data"
        file_list = [
            train_files_path + "/%s" % x for x in os.listdir(train_files_path)
        ]
        train_examples = self.get_example_num(file_list)
        result = {}

        model_path = "word2vec_model"
        dict_path = "./thirdparty/test_build_dict_word_to_id_"
        batch_size = 20000
        start_index = 0
        last_index = 5
        emb_size = 300
        if os.path.exists(model_path):
            # do infer
            args = parse_args()
            test_dir = args.test_dir
            args.model_dir = model_path
            args.batch_size = batch_size
            args.start_index = start_index
            args.last_index = last_index
            args.emb_size = emb_size
            args.dict_path = dict_path
            use_cuda = True if args.use_cuda else False
            print("start index: ", start_index, " last_index:", last_index)
            vocab_size, test_reader, id2word = utils.prepare_data(
                test_dir, dict_path, batch_size=batch_size)
            print("vocab_size:", vocab_size)
            res_dict = infer_epoch(
                args,
                vocab_size,
                test_reader=test_reader,
                use_cuda=use_cuda,
                i2w=id2word)

            acc_result = []
            for epoch in res_dict:
                acc_result.append(res_dict[epoch])
            result['acc'] = max(acc_result)
            result['acc_list'] = acc_result
            file_path = "./benchmark_logs/infer_log"
            with open(file_path, 'w') as f:
                f.writelines(str(result))
                logger.info("write infer log to work path:wirte %s" %
                            (file_path))
            logger.info("Infer log: {}".format(result))

        if context['fleet_mode'] == "PS":
            # 2. Save training speed
            model_dict = context["env"]["phase"][0]
            running_time = context["model"][model_dict["name"]]["running_time"]
            speed = []
            for time in running_time:
                speed.append(train_examples / float(time))

            result['performance'] = np.mean(speed)
            result['performance_list'] = speed

            result['cost_time'] = np.mean(running_time)
            result['cost_time_list'] = running_time

            result['examples'] = train_examples

            file_path = "./benchmark_logs/training_log"
            with open(file_path, 'w') as f:
                f.writelines(str(result))
                logger.info("write training log to work path:wirte %s" %
                            (file_path))
            logger.info("Trainer: {} Training log: {}".format(context[
                'fleet'].worker_index(), result))

        # 3. Stop Worker
        if context['fleet_mode'] == 'PS':
            context['fleet'].stop_worker()
