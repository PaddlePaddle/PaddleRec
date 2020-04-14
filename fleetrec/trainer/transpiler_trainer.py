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

"""
Training use fluid with DistributeTranspiler
"""
import os

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet

from fleetrec.trainer.trainer import Trainer
from fleetrec.utils import envs


class TranspileTrainer(Trainer):
    def __init__(self, config=None):
        Trainer.__init__(self, config)
        self.processor_register()
        self.model = None
        self.inference_models = []
        self.increment_models = []

    def processor_register(self):
        print("Need implement by trainer, `self.regist_context_processor('uninit', self.instance)` must be the first")

    def _get_dataset(self):
        namespace = "train.reader"

        inputs = self.model.inputs()
        threads = envs.get_global_env("train.threads", None)
        batch_size = envs.get_global_env("batch_size", None, namespace)
        reader_class = envs.get_global_env("class", None, namespace)
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        reader = os.path.join(abs_dir, '..', 'reader_implement.py')
        pipe_cmd = "python {} {} {}".format(reader, reader_class, "TRAIN")
        train_data_path = envs.get_global_env("train_data_path", None, namespace)

        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(inputs)
        dataset.set_pipe_command(pipe_cmd)
        dataset.set_batch_size(batch_size)
        dataset.set_thread(threads)
        file_list = [
            os.path.join(train_data_path, x)
            for x in os.listdir(train_data_path)
        ]

        dataset.set_filelist(file_list)
        return dataset

    def save(self, epoch_id, namespace, is_fleet=False):
        def need_save(epoch_id, epoch_interval, is_last=False):
            if is_last:
                return True

            if epoch_id == -1:
                return False

            return epoch_id % epoch_interval == 0

        def save_inference_model():
            save_interval = envs.get_global_env("save.inference.epoch_interval", -1, namespace)

            if not need_save(epoch_id, save_interval, False):
                return

            print("save inference model is not supported now.")
            return

            feed_varnames = envs.get_global_env("save.inference.feed_varnames", None, namespace)
            fetch_varnames = envs.get_global_env("save.inference.fetch_varnames", None, namespace)
            fetch_vars = [fluid.global_scope().vars[varname] for varname in fetch_varnames]
            dirname = envs.get_global_env("save.inference.dirname", None, namespace)

            assert dirname is not None
            dirname = os.path.join(dirname, str(epoch_id))

            if is_fleet:
                fleet.save_inference_model(dirname, feed_varnames, fetch_vars)
            else:
                fluid.io.save_inference_model(dirname, feed_varnames, fetch_vars, self._exe)
            self.inference_models.append((epoch_id, dirname))

        def save_persistables():
            save_interval = envs.get_global_env("save.increment.epoch_interval", -1, namespace)

            if not need_save(epoch_id, save_interval, False):
                return

            dirname = envs.get_global_env("save.increment.dirname", None, namespace)

            assert dirname is not None
            dirname = os.path.join(dirname, str(epoch_id))

            if is_fleet:
                fleet.save_persistables(self._exe, dirname)
            else:
                fluid.io.save_persistables(self._exe, dirname)
            self.increment_models.append((epoch_id, dirname))

        save_persistables()
        save_inference_model()

    def instance(self, context):
        models = envs.get_global_env("train.model.models")
        model_class = envs.lazy_instance(models, "TrainModel")
        self.model = model_class(None)
        context['status'] = 'init_pass'

    def init(self, context):
        print("Need to be implement")
        context['is_exit'] = True

    def train(self, context):
        print("Need to be implement")
        context['is_exit'] = True

    def infer(self, context):
        context['is_exit'] = True

    def terminal(self, context):
        print("clean up and exit")
        context['is_exit'] = True
