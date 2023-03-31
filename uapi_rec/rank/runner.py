# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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

import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from base import BaseRunner


class RankRunner(BaseRunner):
    def train(self, config_path, cli_args, device, ips):
        python = self.distributed(device, ips)
        args = self._gather_opts_args(cli_args)
        args_str = ' '.join(str(arg) for arg in args)
        batch_size_value = 0
        learning_rate_value = 0
        iters_value = 0
        save_dir_value = ""
        device = "cpu"
        try:
            batch_size_value = args_str.split("--device ")[1].split(" ")[0]
        except:
            pass
        try:
            batch_size_value = args_str.split("--batch_size ")[1].split(" ")[0]
        except:
            pass
        try:
            learning_rate_value = args_str.split("--learning_rate ")[1].split(
                " ")[0]
        except:
            pass
        try:
            iters_value = args_str.split("--iters ")[1].split(" ")[0]
        except:
            pass
        try:
            save_dir_value = args_str.split("--save_dir ")[1].split(" ")[0]
        except:
            pass
        paras_cmd = f"-o runner.use_gpu=True runner.train_batch_size={batch_size_value} hyper_parameters.optimizer.learning_rate={learning_rate_value} \
        runner.epochs={iters_value} runner.model_save_path={save_dir_value}"

        paddle_pserver_ip_port = "127.0.0.1:29011"
        #gpus_list = "0,1,2,3,4,5,6,7"
        gpus_list = "0,1"
        if "device" in device:
            gpus_list = device.lstrip("gpu:")
            
        server_echo = "PADDLE WILL START PSERVER 29011"
        trainer_echo = "PADDLE WILL START Trainer 0"
        cmd = f"export FLAGS_LAUNCH_BARRIER=0 && \
                export PADDLE_TRAINER_ID=0 && \
                export PADDLE_PSERVER_NUMS=1 && \
                export PADDLE_TRAINERS=1 && \
                export PADDLE_TRAINERS_NUM=1 && \
                export POD_IP=127.0.0.1 && \
                export PADDLE_PSERVERS_IP_PORT_LIST={paddle_pserver_ip_port} && \
                export FLAGS_selected_gpus={gpus_list} && \
                export TRAINING_ROLE=PSERVER && \
                echo {server_echo} && \
                export PADDLE_PORT=29011 && \
                python3.7 -u tools/static_gpubox_trainer.py -m {config_path} {paras_cmd} && \
                export TRAINING_ROLE=TRAINER && \
                echo {trainer_echo} && \
                python3.7 -u tools/static_gpubox_trainer.py -m {config_path} {paras_cmd}"

        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def evaluate(self, config_path, cli_args, device, ips):
        pass

    def predict(self, config_path, cli_args, device):
        pass

    def export(self, config_path, cli_args, device):
        pass

    def infer(self, config_path, cli_args, device):
        pass

    def compression(self, config_path, train_cli_args, export_cli_args, device,
                    train_save_dir):
        pass

    def _gather_opts_args(self, args):
        # Since `--opts` in PaddleRec does not use `action='append'`
        # We collect and arrange all opts args here
        # e.g.: python tools/train.py --config xxx --opts a=1 c=3 --opts b=2
        # => python tools/train.py --config xxx c=3 --opts a=1 b=2
        # NOTE: This is an inplace operation
        def _is_opts_arg(arg):
            return arg.key.lstrip().startswith('--opts')

        # We note that Python built-in `sorted()` preserves the order (stable)
        args = sorted(args, key=_is_opts_arg)
        found = False
        for arg in args:
            if _is_opts_arg(arg):
                if found:
                    arg.key = arg.key.replace('--opts', '')
                else:
                    # Found first
                    found = True

        return args


def _extract_eval_metrics(stdout):
    import re

    _DP = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    pattern = r'Images: \d+ mIoU: (_dp) Acc: (_dp) Kappa: (_dp) Dice: (_dp)'.replace(
        '_dp', _DP)
    keys = ['miou', 'acc', 'kappa', 'dice']

    metric_dict = dict()
    pattern = re.compile(pattern)
    # TODO: Use lazy version to make it more efficient
    lines = stdout.splitlines()
    for line in lines:
        match = pattern.search(line)
        if match:
            for k, v in zip(keys, map(float, match.groups())):
                metric_dict[k] = v
    return metric_dict
