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

import os
import shutil
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TOOLS_PATH = os.path.join(LOCAL_PATH, "..", "..", "tools")
sys.path.append(TOOLS_PATH)

from paddlerec.tools.tools import download_file_and_uncompress, download_file

if __name__ == '__main__':
    url_train = "https://paddlerec.bj.bcebos.com/xdeepfm%2Ftr"
    url_test = "https://paddlerec.bj.bcebos.com/xdeepfm%2Fev"

    train_dir = "train_data"
    test_dir = "test_data"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    print("download and extract starting...")
    download_file(url_train, "./train_data/tr", True)
    download_file(url_test, "./test_data/ev", True)
    print("download and extract finished")

    print("done")
