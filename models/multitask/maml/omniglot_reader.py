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
from paddle.io import IterableDataset
import cv2
import os


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.config = config
        self.n_way = 5
        self.k_spt = 1
        self.k_query = 15
        self.imgsize = 28
        np.random.seed(12345)
        character_folders = [
            os.path.join(family, character) for family in self.file_list
            if os.path.isdir(family) for character in os.listdir(family)
        ]
        imgs_list = []
        for char_fold in character_folders:
            char_list = []
            for file in [
                    os.path.join(char_fold, f) for f in os.listdir(char_fold)
            ]:
                img = cv2.imread(file)
                img = cv2.resize(img, (28, 28))
                img = np.transpose(img, (2, 0, 1))
                img = img[0].astype('float32')  # 只取零通道
                img = img / 255.0
                img = img * 2.0 - 1.0
                char_list.append(img)
            char_list = np.array(char_list)
            imgs_list.append(char_list)
        self.train_imgs = np.array(imgs_list)
        self.train_imgs = self.train_imgs[:, :, np.newaxis, :, :]
        #print('The shape of self.train_imgs: {}'.format(self.train_imgs.shape))   # [973,20,1,28,28]

    def __iter__(self):
        full_lines = []
        self.data = []
        for i in range(3200):
            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(
                self.train_imgs.shape[0], self.n_way, replace=False)
            for j, cur_class in enumerate(selected_cls):
                selected_img = np.random.choice(
                    20, self.k_spt + self.k_query, replace=False)
                # 构造support集和query集
                x_spt.append(self.train_imgs[cur_class][
                    selected_img[:self.k_spt]])
                x_qry.append(self.train_imgs[cur_class][selected_img[
                    self.k_spt:]])
                y_spt.append([j for _ in range(self.k_spt)])
                y_qry.append([j for _ in range(self.k_query)])

            perm = np.random.permutation(self.n_way * self.k_spt)
            x_spt = np.array(x_spt).reshape(
                self.n_way * self.k_spt, 1, self.imgsize,
                self.imgsize)[perm]  # [5,1,1,28,28]=>[5,1,28,28]
            y_spt = np.array(y_spt).reshape(self.n_way *
                                            self.k_spt)[perm]  # [5,1]=>[5,]
            perm = np.random.permutation(self.n_way * self.k_query)
            x_qry = np.array(x_qry).reshape(
                self.n_way * self.k_query, 1, self.imgsize,
                self.imgsize)[perm]  # [5,15,1,28,28]=>[75,1,28,28]
            y_qry = np.array(y_qry).reshape(
                self.n_way * self.k_query)[perm]  # [5,15]=>[75,]

            output_list = []
            output_list.append(np.array(x_spt).astype("float32"))
            output_list.append(np.array(y_spt).astype("int64"))
            output_list.append(np.array(x_qry).astype("float32"))
            output_list.append(np.array(y_qry).astype("int64"))
            yield output_list
