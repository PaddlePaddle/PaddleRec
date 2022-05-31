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
        for file in self.file_list:
            with open(file, "rb") as rf:
                data_train_stage1 = pickle.load(rf)[self.all_col]
                n_samples = data_train_stage1.shape[0]

                aid_set = list(set(data_train_stage1.aid))
                data_train = data_train_stage1

                n_batch = int(np.ceil(n_samples / self.batchsize)) #总量除以batchsize * task_count

                list_prob = []
                for aid in aid_set:
                    list_prob.append(data_train_stage1[data_train_stage1.aid == aid].shape[0])
                list_prob_sum = sum(list_prob)
                for i in range(len(list_prob)):
                    list_prob[i] = list_prob[i] / list_prob_sum
    
                for i_batch in range(n_batch):
                    batch_aid_set = np.random.choice(aid_set, size=self.task_count, replace=True, p=list_prob) # size=task_count
                    list_sup_x, list_sup_y, list_qry_x, list_qry_y = list(), list(), list(), list()

                    for aid in batch_aid_set:
                        batch_sup = data_train[data_train.aid == aid].sample(self.batchsize)
                        batch_qry = data_train[data_train.aid == aid].sample(self.batchsize)

                        batch_sup_x = []
                        batch_sup_x.append(np.array(batch_sup[self.static_context_col])[:]) #[batchsize,7]

                        # sup中dynamic部分
                        temp_list = list()
                        for k in range(len(self.dynamic_context_col)):
                            dy_np = np.array(batch_sup[self.dynamic_context_col[k]])[:]
                            dy_np = np.vstack(dy_np)
                            temp_list.append(dy_np) 
                        temp_np = np.concatenate(temp_list,axis=1)
    
                        batch_sup_x.append(temp_np) #[batchsize,50]
                        batch_sup_x.append(np.array(batch_sup[self.col_length_name])[:])#[batchsize,7]
                        batch_sup_x.append(np.array(batch_sup[self.ad_col])[:])#[batchsize,6]
    
                        batch_sup_y = np.array(batch_sup[self.label_col].values)[:]#[batchsize,1]

                        batch_qry_x = []
                        batch_qry_x.append(np.array(batch_qry[self.static_context_col])[:])#[batchsize,7]

                        # qry中dynamic部分
                        temp_list = list()
                        for k in range(len(self.dynamic_context_col)):
                            dy_np = np.array(batch_qry[self.dynamic_context_col[k]])[:]
                            dy_np = np.vstack(dy_np)
                            temp_list.append(dy_np) 
                        temp_np = np.concatenate(temp_list,axis=1)

                        batch_qry_x.append(temp_np)#[batchsize,50]
                        batch_qry_x.append(np.array(batch_qry[self.col_length_name])[:])#[batchsize,7]
                        batch_qry_x.append(np.array(batch_qry[self.ad_col])[:])#[batchsize,6]

                        batch_qry_y = np.array(batch_qry[self.label_col].values)[:]#[batchsize,1]

                        list_sup_x.append(batch_sup_x)# shape = [5,batchsize,7+50+7+6]
                        list_sup_y.append(batch_sup_y)# shape = [5,batchsize,1]
                        list_qry_x.append(batch_qry_x)# shape = [5,batchsize,7+50+7+6]
                        list_qry_y.append(batch_qry_y)# shape = [5,batchsize,1]

                    output_list = []
                    output_list.append(list_sup_x)
                    output_list.append(list_sup_y)
                    output_list.append(list_qry_x)
                    output_list.append(list_qry_y)

                    yield output_list
