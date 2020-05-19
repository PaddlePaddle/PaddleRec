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

from paddlerec.core.trainer import Trainer
from paddlerec.core.utils import envs
from paddlerec.core.utils import dataloader_instance


class TranspileTrainer(Trainer):
    def __init__(self, config=None):
        Trainer.__init__(self, config)
        device = envs.get_global_env("train.device", "cpu")
        if device == 'gpu':
            self._place = fluid.CUDAPlace(0)
            self._exe = fluid.Executor(self._place)
        self.processor_register()
        self.model = None
        self.inference_models = []
        self.increment_models = []

    def processor_register(self):
        print("Need implement by trainer, `self.regist_context_processor('uninit', self.instance)` must be the first")

    def _get_dataloader(self, state="TRAIN"):
        if state == "TRAIN":
            dataloader = self.model._data_loader
            namespace = "train.reader"
            class_name = "TrainReader"
        else:
            dataloader = self.model._infer_data_loader
            namespace = "evaluate.reader"
            class_name = "EvaluateReader"

        batch_size = envs.get_global_env("batch_size", None, namespace)
        reader_class = envs.get_global_env("class", None, namespace)
        print("batch_size: {}".format(batch_size))
        reader = dataloader_instance.dataloader(
            reader_class, state, self._config_yaml)

        reader_class = envs.lazy_instance_by_fliename(reader_class, class_name)
        reader_ins = reader_class(self._config_yaml)
        if hasattr(reader_ins, 'generate_batch_from_trainfiles'):
            dataloader.set_sample_list_generator(reader)
        else:
            dataloader.set_sample_generator(reader, batch_size)

        debug_mode = envs.get_global_env("reader_debug_mode", False, namespace)
        if debug_mode:
            print("--- DataLoader Debug Mode Begin , show pre 10 data ---")
            for idx, line in enumerate(reader()):
                print(line)
                if idx >= 9:
                    break
            print("--- DataLoader Debug Mode End , show pre 10 data ---")
            exit(0)
        return dataloader

    def _get_dataset_ins(self):
        count = 0
        for f in self.files:
            for _, _ in enumerate(open(f, 'r')):
                count += 1
        return count

    def _get_dataset(self, state="TRAIN"):
        if state == "TRAIN":
            inputs = self.model.get_inputs()
            namespace = "train.reader"
            train_data_path = envs.get_global_env(
                "train_data_path", None, namespace)
        else:
            inputs = self.model.get_infer_inputs()
            namespace = "evaluate.reader"
            train_data_path = envs.get_global_env(
                "test_data_path", None, namespace)

        threads = int(envs.get_runtime_environ("train.trainer.threads"))
        batch_size = envs.get_global_env("batch_size", None, namespace)
        reader_class = envs.get_global_env("class", None, namespace)
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        reader = os.path.join(abs_dir, '../utils', 'dataset_instance.py')
        pipe_cmd = "python {} {} {} {}".format(
            reader, reader_class, state, self._config_yaml)

        if train_data_path.startswith("paddlerec::"):
            package_base = envs.get_runtime_environ("PACKAGE_BASE")
            assert package_base is not None
            train_data_path = os.path.join(
                package_base, train_data_path.split("::")[1])

        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_use_var(inputs)
        dataset.set_pipe_command(pipe_cmd)
        dataset.set_batch_size(batch_size)
        dataset.set_thread(threads)
        file_list = [
            os.path.join(train_data_path, x)
            for x in os.listdir(train_data_path)
        ]
        self.files = file_list
        dataset.set_filelist(self.files)

        debug_mode = envs.get_global_env("reader_debug_mode", False, namespace)
        if debug_mode:
            print(
                "--- Dataset Debug Mode Begin , show pre 10 data of {}---".format(file_list[0]))
            os.system("cat {} | {} | head -10".format(file_list[0], pipe_cmd))
            print(
                "--- Dataset Debug Mode End , show pre 10 data of {}---".format(file_list[0]))
            exit(0)

        return dataset

    def save(self, epoch_id, namespace, is_fleet=False):
        def need_save(epoch_id, epoch_interval, is_last=False):
            if is_last:
                return True

            if epoch_id == -1:
                return False

            return epoch_id % epoch_interval == 0

        def save_inference_model():
            save_interval = envs.get_global_env(
                "save.inference.epoch_interval", -1, namespace)

            if not need_save(epoch_id, save_interval, False):
                return

            #  print("save inference model is not supported now.")
            #  return

            feed_varnames = envs.get_global_env(
                "save.inference.feed_varnames", None, namespace)
            fetch_varnames = envs.get_global_env(
                "save.inference.fetch_varnames", None, namespace)
            if feed_varnames is None or fetch_varnames is None:
                return

            fetch_vars = [fluid.default_main_program().global_block().vars[varname]
                          for varname in fetch_varnames]
            dirname = envs.get_global_env(
                "save.inference.dirname", None, namespace)

            assert dirname is not None
            dirname = os.path.join(dirname, str(epoch_id))

            if is_fleet:
                fleet.save_inference_model(
                    self._exe, dirname, feed_varnames, fetch_vars)
            else:
                fluid.io.save_inference_model(
                    dirname, feed_varnames, fetch_vars, self._exe)
            self.inference_models.append((epoch_id, dirname))

        def save_persistables():
            save_interval = envs.get_global_env(
                "save.increment.epoch_interval", -1, namespace)

            if not need_save(epoch_id, save_interval, False):
                return

            dirname = envs.get_global_env(
                "save.increment.dirname", None, namespace)

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
        model_class = envs.lazy_instance_by_fliename(models, "Model")
        self.model = model_class(None)
        context['status'] = 'init_pass'

    def init(self, context):
        print("Need to be implement")
        context['is_exit'] = True

    def dataloader_train(self, context):
        print("Need to be implement")
        context['is_exit'] = True

    def dataset_train(self, context):
        print("Need to be implement")
        context['is_exit'] = True

    def infer(self, context):
        infer_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(infer_program, startup_program):
                self.model.infer_net()

        if self.model._infer_data_loader is None:
            context['status'] = 'terminal_pass'
            return

        reader = self._get_dataloader("Evaluate")

        metrics_varnames = []
        metrics_format = []

        metrics_format.append("{}: {{}}".format("epoch"))
        metrics_format.append("{}: {{}}".format("batch"))

        for name, var in self.model.get_infer_results().items():
            metrics_varnames.append(var.name)
            metrics_format.append("{}: {{}}".format(name))

        metrics_format = ", ".join(metrics_format)
        self._exe.run(startup_program)

        model_list = self.increment_models

        evaluate_only = envs.get_global_env(
            'evaluate_only', False, namespace='evaluate')
        if evaluate_only:
            model_list = [(0, envs.get_global_env(
                'evaluate_model_path', "", namespace='evaluate'))]

        is_return_numpy = envs.get_global_env(
            'is_return_numpy', True, namespace='evaluate')

        for (epoch, model_dir) in model_list:
            print("Begin to infer No.{} model, model_dir: {}".format(
                epoch, model_dir))
            program = infer_program.clone()
            fluid.io.load_persistables(self._exe, model_dir, program)
            reader.start()
            batch_id = 0
            try:
                while True:
                    metrics_rets = self._exe.run(
                        program=program,
                        fetch_list=metrics_varnames,
                        return_numpy=is_return_numpy)

                    metrics = [epoch, batch_id]
                    metrics.extend(metrics_rets)

                    if batch_id % 2 == 0 and batch_id != 0:
                        print(metrics_format.format(*metrics))
                    batch_id += 1
            except fluid.core.EOFException:
                reader.reset()

        context['status'] = 'terminal_pass'

    def terminal(self, context):
        print("clean up and exit")
        context['is_exit'] = True
