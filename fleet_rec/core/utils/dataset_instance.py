#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys

from fleetrec.core.utils.envs import lazy_instance_by_fliename

if len(sys.argv) != 4:
    raise ValueError("reader only accept 3 argument: 1. reader_class 2.train/evaluate 3.yaml_abs_path")

reader_package = sys.argv[1]

if sys.argv[2] == "TRAIN":
    reader_name = "TrainReader"
else:
    reader_name = "EvaluateReader"

yaml_abs_path = sys.argv[3]
reader_class = lazy_instance_by_fliename(reader_package, reader_name)
reader = reader_class(yaml_abs_path)
reader.init()
reader.run_from_stdin()
