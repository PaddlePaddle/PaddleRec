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

from fleetrec.core.reader import Reader
from fleetrec.core.utils import envs
from collections import defaultdict
import numpy as np


class TrainReader(Reader):
    def init(self):
        all_field_id = ['101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124', '125', '126', '127', '128', '129',
                        '205', '206', '207', '210', '216', '508', '509', '702', '853', '301']
        self.all_field_id_dict = defaultdict(int)
        for i,field_id in enumerate(all_field_id):
            self.all_field_id_dict[field_id] = [False,i]

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
            features = line.strip().split(',')
            #ctr = list(map(int, features[1]))
            #cvr = list(map(int, features[2]))
            ctr = int(features[1])
            cvr = int(features[2])
            
            padding = 0
            output = [(field_id,[]) for field_id in self.all_field_id_dict]

            for elem in features[4:]:
                field_id,feat_id = elem.strip().split(':')
                if field_id not in self.all_field_id_dict:
                    continue
                self.all_field_id_dict[field_id][0] = True
                index = self.all_field_id_dict[field_id][1]
                #feat_id = list(map(int, feat_id))    
                output[index][1].append(int(feat_id)) 
                
            for field_id in self.all_field_id_dict:
                visited,index = self.all_field_id_dict[field_id]
                if visited:
                    self.all_field_id_dict[field_id][0] = False
                else:
                    output[index][1].append(padding) 
            output.append(('ctr', [ctr]))
            output.append(('cvr', [cvr]))
            yield output
        return reader
