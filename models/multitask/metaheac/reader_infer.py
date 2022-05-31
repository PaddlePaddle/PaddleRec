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
import numpy as np
import pickle
import paddle

from collections import defaultdict
from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.config = config
        self.task_count = config.get("hyper_parameters.task_count")
        self.batchsize = config.get("hyper_parameters.batch_size")

        self.static_context_col = ['carrier', 'consumptionAbility', 'LBS', 
        'age','education', 'gender', 'house']
        self.dynamic_context_col = ['interest1', 'interest2', 'interest3', 
        'kw1', 'kw2', 'topic1', 'topic2']
        self.ad_col = ['advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 
        'productId', 'productType']
        self.col_length_name = [x + '_length' for x in self.dynamic_context_col]
        self.label_col = 'label'
        self.train_col = self.static_context_col + self.dynamic_context_col + self.col_length_name + self.ad_col 
        self.all_col = [self.label_col, 'aid'] + self.static_context_col + self.dynamic_context_col +  self.col_length_name + self.ad_col


    def __iter__(self):
        np.random.seed(2021)
        self.file_list.sort()
        print(self.file_list)

        with open(self.file_list[0], "rb") as data_test_stage1, open(self.file_list[1], "rb") as data_test_stage2:
            data_test_stage1 = pickle.load(data_test_stage1)[self.all_col]
            data_test_stage2 = pickle.load(data_test_stage2)[self.all_col]

            aid_set = set(data_test_stage1.aid)

            for aid in aid_set:
                task_test_stage1 = data_test_stage1[data_test_stage1.aid == aid]
                task_test_stage2 = data_test_stage2[data_test_stage2.aid == aid]

                data_train = task_test_stage1.sample(frac=1)
                data_test = task_test_stage2

                output_list = list()

                batch_sup_x = []
                batch_sup_x.append(np.array(data_train[self.static_context_col])[:]) #shape=[*, 7]

                # data_stage1中dynamic部分
                temp_list = list()
                for k in range(len(self.dynamic_context_col)):
                    dy_np = np.array(data_train[self.dynamic_context_col[k]])[:]
                    dy_np = np.vstack(dy_np)
                    temp_list.append(dy_np) 
                temp_np = np.concatenate(temp_list, axis=1)
                batch_sup_x.append(temp_np) #shape = [*, 50]

                batch_sup_x.append(np.array(data_train[self.col_length_name])[:])#shape = [*,7]
                batch_sup_x.append(np.array(data_train[self.ad_col])[:])#shape = [*,6]

                batch_sup_y = np.array(data_train[self.label_col].values)[:]#shape = [*,1]   
                
                batch_qry_x = []
                batch_qry_x.append(np.array(data_test[self.static_context_col])[:]) #shape=[*, 7]

                # data_stage2中dynamic部分
                temp_list = list()
                for k in range(len(self.dynamic_context_col)):
                    dy_np = np.array(data_test[self.dynamic_context_col[k]])[:]
                    dy_np = np.vstack(dy_np)
                    temp_list.append(dy_np)
                temp_np = np.concatenate(temp_list, axis=1) 
                batch_qry_x.append(temp_np) #shape = [*, 50]

                batch_qry_x.append(np.array(data_test[self.col_length_name])[:])#shape = [*,7]
                batch_qry_x.append(np.array(data_test[self.ad_col])[:])#shape = [*,6]

                batch_qry_y = np.array(data_test[self.label_col].values)[:]#shape = [*,1]   

                output_list.append(np.array([aid])) #本次子任务的aid
                output_list.append(batch_sup_x)# shape = [*, 7+50+7+6]
                output_list.append(batch_sup_y)# shape = [*, 1]
                output_list.append(batch_qry_x)# shape = [*, 7+50+7+6]
                output_list.append(batch_qry_y)# shape = [*, 1]

                yield output_list        
