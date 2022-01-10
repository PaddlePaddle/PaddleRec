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
import random
from collections import defaultdict

from paddle.io import IterableDataset


class Rating(object):
    def __init__(self, trainingSet, testSet):
        self.evalSettings = {"cv": 5, "b": 1}  # "-cv 5 -b 1"
        self.user = {}  # map user names to id
        self.item = {}  # map item names to id
        self.id2user = {}
        self.id2item = {}
        self.userMeans = {}  # mean values of users's ratings
        self.itemMeans = {}  # mean values of items's ratings
        self.globalMean = 0
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(
            dict)  # test set in the form of [user][item]=rating
        self.testSet_i = defaultdict(
            dict)  # test set in the form of [item][user]=rating
        self.rScale = []  # rating scale
        self.trainingData = trainingSet[:]
        self.testData = testSet[:]
        self.__generateSet()
        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()

    def __generateSet(self):
        scale = set()

        for i, entry in enumerate(self.trainingData):
            userName, itemName, rating = entry
            # makes the rating within the range [0, 1].
            # rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            # self.trainingData[i][2] = rating
            # order the user
            if userName not in self.user:
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # order the item
            if itemName not in self.item:
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
                # userList.append
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating
            scale.add(float(rating))
        self.rScale = list(scale)
        self.rScale.sort()
        for entry in self.testData:
            userName, itemName, rating = entry
            self.testSet_u[userName][itemName] = rating
            self.testSet_i[itemName][userName] = rating

    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total == 0:
            self.globalMean = 0
        else:
            self.globalMean = total / len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user:
            self.userMeans[u] = sum(self.trainSet_u[u].values()) / len(
                self.trainSet_u[u])

    def __computeItemMean(self):
        for c in self.item:
            self.itemMeans[c] = sum(self.trainSet_i[c].values()) / len(
                self.trainSet_i[c])

    def getUserId(self, u):
        if u in self.user:
            return self.user[u]

    def getItemId(self, i):
        if i in self.item:
            return self.item[i]

    def trainingSize(self):
        return len(self.user), len(self.item), len(self.trainingData)

    def testSize(self):
        return len(self.testSet_u), len(self.testSet_i), len(self.testData)

    def contains(self, u, i):
        if u in self.user and i in self.trainSet_u[u]:
            return True
        else:
            return False

    def containsUser(self, u):
        if u in self.user:
            return True
        else:
            return False

    def containsItem(self, i):
        if i in self.item:
            return True
        else:
            return False

    def userRated(self, u):
        return list(self.trainSet_u[u].keys()), list(self.trainSet_u[u].values(
        ))

    def itemRated(self, i):
        return list(self.trainSet_i[i].keys()), list(self.trainSet_i[i].values(
        ))

    def row(self, u):
        k, v = self.userRated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        k, v = self.itemRated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.userRated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m

    def sRow(self, u):
        return self.trainSet_u[u]

    def sCol(self, c):
        return self.trainSet_i[c]

    def rating(self, u, c):
        if self.contains(u, c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return self.rScale[0], self.rScale[1]

    def elemCount(self):
        return len(self.trainingData)


def loadDataSet(file, bTest=False, binarized=False, threshold=3.0):
    trainingData, testData = [], []

    with open(file) as f:
        ratings = f.readlines()

    order = ["0", "1", "2"]

    for lineNo, line in enumerate(ratings):
        items = line.strip().split("\t")
        try:
            userId = items[int(order[0])]
            itemId = items[int(order[1])]
            rating = items[int(order[2])]

            if binarized:
                if float(items[int(order[2])]) < threshold:
                    continue
                else:
                    rating = 1

        except ValueError:
            print("Error! Dataset")

        if bTest:
            testData.append([userId, itemId, float(rating)])
        else:
            trainingData.append([userId, itemId, float(rating)])
    if bTest:
        return testData
    else:
        return trainingData


def loadRelationship(file):
    relation = []
    with open(file) as f:
        relations = f.readlines()

    order = ["0", "1"]
    for lineNo, line in enumerate(relations):
        items = line.strip().split("\t")
        userId1 = items[int(order[0])]
        userId2 = items[int(order[1])]
        weight = 1

        relation.append([userId1, userId2, weight])

    return relation


def crossValidation(data, k, binarized=False):
    if k <= 1 or k > 10:
        k = 3
    for i in range(k):
        trainingSet = []
        testSet = []
        for ind, line in enumerate(data):
            if ind % k == i:
                if binarized:
                    if line[2]:
                        testSet.append(line[:])
                else:
                    testSet.append(line[:])
            else:
                trainingSet.append(line[:])
        yield trainingSet, testSet


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.data = Rating(trainingSet, testSet)

    def __iter__(self):
        full_lines = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for line in rf:
                    output_list = []
                    features = line.strip().split(',')
                    user_input = [int(features[0])]
                    item_input = [int(features[1])]
                    label = [int(features[2])]
                    output_list.append(np.array(user_input).astype('int64'))
                    output_list.append(np.array(item_input).astype('int64'))
                    output_list.append(np.array(label).astype('int64'))
                    yield output_list
