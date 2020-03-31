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
import os
import time
import numpy as np
import logging
import paddle.fluid as fluid
from network import CTR
from argument import params_args

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def get_dataset(inputs, params):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python dataset_generator.py")
    dataset.set_batch_size(params.batch_size)
    dataset.set_thread(int(params.cpu_num))
    file_list = [
        str(params.train_files_path) + "/%s" % x
        for x in os.listdir(params.train_files_path)
    ]
    dataset.set_filelist(file_list)
    logger.info("file list: {}".format(file_list))
    return dataset


def train(params):
    ctr_model = CTR()
    inputs = ctr_model.input_data(params)
    avg_cost, auc_var, batch_auc_var = ctr_model.net(inputs, params)
    optimizer = fluid.optimizer.Adam(params.learning_rate)
    optimizer.minimize(avg_cost)
    fluid.default_main_program()
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    dataset = get_dataset(inputs, params)

    logger.info("Training Begin")
    for epoch in range(params.epochs):
        start_time = time.time()
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset,
                               fetch_list=[auc_var],
                               fetch_info=["Epoch {} auc ".format(epoch)],
                               print_period=100,
                               debug=False)
        end_time = time.time()
        logger.info("epoch %d finished, use time=%d\n" %
                    ((epoch), end_time - start_time))

        if params.test:
            model_path = (str(params.model_path) + "/" + "epoch_" + str(epoch))
            fluid.io.save_persistables(executor=exe, dirname=model_path)

    logger.info("Train Success!")


if __name__ == "__main__":
    params = params_args()
    train(params)