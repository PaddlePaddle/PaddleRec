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

"---- network config -----"
with_shuffle_batch = False
neg_num = 5
sparse_feature_number = 354051
sparse_feature_dim = 300

"---- training config -----"
epochs = 1
batch_size = 1000
learning_rate = 1e-4
decay_steps = 100000
decay_rate = 0.999
train_data_path = "./train_data"
word_count_dict_path = "./"

"---- framework config -----"
reader_type = "DataLoader"  # DataLoader / QueueDataset
pipe_command = "python ctr_reader.py"
sync_mode = "async"  # sync / async /geo / heter
geo_step = 400
thread_num = 16
use_cuda = True
split_file_list = False

"---- profiler config -----"
print_period = 100
dataset_debug = False

"---- model load & save config -----"
warmup_model_path = None
