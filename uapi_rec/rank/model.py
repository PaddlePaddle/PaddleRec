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
from base import BaseModel
from base.utils.cache import get_cache_dir
from base.utils.arg import CLIArgument
from base.utils.misc import abspath
from base.utils import stagelog


class RankModel(BaseModel):
    #_DUMMY_DATASET_DIR = os.path.join(get_cache_dir(), 'ppseg_dummy_dataset')

    def train(self,
              dataset=None,
              batch_size=None,
              learning_rate=None,
              epochs_iters=None,
              ips=None,
              device='gpu',
              resume_path=None,
              dy2st=False,
              amp='OFF',
              use_vdl=True,
              save_dir=None):
        if dataset is not None:
            # NOTE: We must use an absolute path here,
            # so we can run the scripts either inside or outside the repo dir.
            dataset = abspath(dataset)
        if resume_path is not None:
            resume_path = abspath(resume_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join('output', 'train'))

        # Update YAML config file
        config = self.config.copy()
        if dataset is not None:
            config.update_dataset(dataset)
        if dy2st:
            config._update_dy2st(dy2st)
        config_path = self._config_path()
        #config.dump(config_path)

        # Parse CLI arguments
        cli_args = []
        if batch_size is not None:
            cli_args.append(CLIArgument('--batch_size', batch_size))
        if learning_rate is not None:
            cli_args.append(CLIArgument('--learning_rate', learning_rate))
        if epochs_iters is not None:
            cli_args.append(CLIArgument('--iters', epochs_iters))
        if device is not None:
            device_type, _ = self.runner.parse_device(device)
            cli_args.append(CLIArgument('--device', device_type))
        if resume_path is not None:
            model_dir = os.path.dirname(resume_path)
            cli_args.append(CLIArgument('--resume_path', model_dir))
        if amp is not None:
            if amp != 'OFF':
                cli_args.append(CLIArgument('--precision', 'fp16'))
                cli_args.append(CLIArgument('--amp_level', amp))
        if use_vdl:
            cli_args.append(CLIArgument('--use_vdl', '', sep=''))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        with stagelog.StageLogTrain(
                learning_rate=learning_rate
                if learning_rate is not None else config._get_learning_rate(),
                epoch_iters=epochs_iters
                if epochs_iters is not None else config._get_epochs_iters(),
                batch_size=batch_size
                if batch_size is not None else config._get_batch_size(),
                data_path=config.train_dataset['dataset_root'],
                yaml_path=config_path,
                vdl_path=save_dir,
                save_dir=save_dir):
            return self.runner.train(config_path, cli_args, device, ips)

    def evaluate(self,
                 weight_path,
                 dataset=None,
                 batch_size=None,
                 ips=None,
                 device='gpu',
                 amp='OFF'):
        pass

    def predict(self, weight_path, input_path, device='gpu', save_dir=None):
        pass

    def export(self, weight_path, save_dir=None, input_shape=None):
        pass

    def infer(self, model_dir, input_path, device='gpu', save_dir=None):
        pass

    def _config_path(self):
        return self.model_info['config_path']

    def compression(self,
                    weight_path,
                    dataset=None,
                    batch_size=None,
                    learning_rate=None,
                    epochs_iters=None,
                    device='gpu',
                    use_vdl=True,
                    save_dir=None,
                    input_shape=None):
        pass

    def _create_dummy_dataset(self):
        # Create a PaddleRec-style dataset
        # We will use this fake dataset to pass the config checks of PaddleRec

        dir_ = os.path.abspath(self._DUMMY_DATASET_DIR)
        if os.path.exists(dir_):
            return dir_
        else:
            os.makedirs(dir_)
            fake_im_filename = 'fake_im.jpg'
            fake_mask_filename = 'fake_mask.png'
            with open(os.path.join(dir_, 'train.txt'), 'w') as f:
                f.write(f'{fake_im_filename} {fake_mask_filename}')
            with open(os.path.join(dir_, 'val.txt'), 'w') as f:
                f.write(f'{fake_im_filename} {fake_mask_filename}')
            return dir_
