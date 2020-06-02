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

from paddlerec.core.reader import Reader


class TrainReader(Reader):
    def init(self):
        pass

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary

        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            l = line.strip().split(',')
            l = list(map(float, l))
            label_income = []
            label_marital = []
            data = l[2:]
            if int(l[1]) == 0:
                label_income = [1, 0]
            elif int(l[1]) == 1:
                label_income = [0, 1]
            if int(l[0]) == 0:
                label_marital = [1, 0]
            elif int(l[0]) == 1:
                label_marital = [0, 1]
            # label_income = np.array(label_income)
            # label_marital = np.array(label_marital)
            feature_name = ["input", "label_income", "label_marital"]
            yield zip(feature_name, [data] + [label_income] + [label_marital])

        return reader
