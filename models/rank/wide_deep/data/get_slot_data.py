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
import yaml
from paddlerec.core.reader import Reader
from paddlerec.core.utils import envs
try:
    import cPickle as pickle
except ImportError:
    import pickle
import paddle.fluid.incubate.data_generator as dg

class TrainReader(dg.MultiSlotDataGenerator):
    def __init__(self, config):
        dg.MultiSlotDataGenerator.__init__(self)

        if os.path.isfile(config):
            with open(config, 'r') as rb:
                _config = yaml.load(rb.read(), Loader=yaml.FullLoader)
        else:
            raise ValueError("reader config only support yaml")

    def init(self):
        pass

    def _process_line(self, line):
        line = line.strip().split(',')
        features = list(map(float, line))
        wide_feat = features[0:8]
        deep_feat = features[8:58 + 8]
        label = features[-1]
        return wide_feat, deep_feat, [label]

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def data_iter():
            wide_feat, deep_deat, label = self._process_line(line)

            s = ""
            for i in [('wide_input', wide_feat), ('deep_input', deep_deat), ('label', label)]:
                k = i[0]
                v = i[1]
                for j in v:
                    s += " " + k + ":" + str(j)
            print s.strip()
            yield None

        return data_iter

reader = TrainReader("../config.yaml")
reader.init()
reader.run_from_stdin()
