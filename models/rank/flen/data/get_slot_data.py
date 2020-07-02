#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.incubate.data_generator as dg


class CriteoDataset(dg.MultiSlotDataGenerator):
    """
    DacDataset: inheritance MultiSlotDataGeneratior, Implement data reading
    Help document: http://wiki.baidu.com/pages/viewpage.action?pageId=728820675
    """

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            features = line.strip().split(',')

            label = [int(features[0])]

            s = "click:" + str(label[0])
            for i, elem in enumerate(features[1:13]):
                s += " user_" + str(i) + ":" + str(elem)
            for i, elem in enumerate(features[13:16]):
                s += " item_" + str(i) + ":" + str(elem)
            for i, elem in enumerate(features[16:]):
                s += " contex_" + str(i) + ":" + str(elem)
            print(s.strip())
            yield None

        return reader


d = CriteoDataset()
d.run_from_stdin()
