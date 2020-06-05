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
    url = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"
    url2 = "https://paddlerec.bj.bcebos.com/deepfm%2Ffeat_dict_10.pkl2"

    print("download and extract starting...")
    download_file_and_uncompress(url)
    download_file(url2, "./sample_data/feat_dict_10.pkl2", True)
    print("download and extract finished")

    print("preprocessing...")
    os.system("python preprocess.py")
    print("preprocess done")

    shutil.rmtree("raw_data")
    print("done")
