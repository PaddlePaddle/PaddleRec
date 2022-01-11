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
from random import shuffle, randint, choice
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


class SparseMatrix():
    'matrix used to store raw data'

    def __init__(self, triple):
        self.matrix_User = {}
        self.matrix_Item = {}
        for item in triple:
            if item[0] not in self.matrix_User:
                self.matrix_User[item[0]] = {}
            if item[1] not in self.matrix_Item:
                self.matrix_Item[item[1]] = {}
            self.matrix_User[item[0]][item[1]] = item[2]
            self.matrix_Item[item[1]][item[0]] = item[2]
        self.elemNum = len(triple)
        self.size = (len(self.matrix_User), len(self.matrix_Item))

    def sRow(self, r):
        if r not in self.matrix_User:
            return {}
        else:
            return self.matrix_User[r]

    def sCol(self, c):
        if c not in self.matrix_Item:
            return {}
        else:
            return self.matrix_Item[c]

    def row(self, r):
        if r not in self.matrix_User:
            return np.zeros((1, self.size[1]))
        else:
            array = np.zeros((1, self.size[1]))
            ind = list(self.matrix_User[r].keys())
            val = list(self.matrix_User[r].values())
            array[0][ind] = val
            return array

    def col(self, c):
        if c not in self.matrix_Item:
            return np.zeros((1, self.size[0]))
        else:
            array = np.zeros((1, self.size[0]))
            ind = list(self.matrix_Item[c].keys())
            val = list(self.matrix_Item[c].values())
            array[0][ind] = val
            return array

    def elem(self, r, c):
        if not self.contains(r, c):
            return 0
        return self.matrix_User[r][c]

    def contains(self, r, c):
        if r in self.matrix_User and c in self.matrix_User[r]:
            return True
        return False

    def elemCount(self):
        return self.elemNum

    def size(self):
        return self.size


class Social(object):
    def __init__(self, relation=None):
        self.user = {}  # used to store the order of users
        self.relation = relation
        self.followees = defaultdict(dict)
        self.followers = defaultdict(dict)
        self.trustMatrix = self.__generateSet()

    def __generateSet(self):
        triple = []
        for line in self.relation:
            userId1, userId2, weight = line
            # add relations to dict
            self.followees[userId1][userId2] = weight
            self.followers[userId2][userId1] = weight
            # order the user
            if userId1 not in self.user:
                self.user[userId1] = len(self.user)
            if userId2 not in self.user:
                self.user[userId2] = len(self.user)
            triple.append([self.user[userId1], self.user[userId2], weight])
        return SparseMatrix(triple)

    def row(self, u):
        # return user u's followees
        return self.trustMatrix.row(self.user[u])

    def col(self, u):
        # return user u's followers
        return self.trustMatrix.col(self.user[u])

    def elem(self, u1, u2):
        return self.trustMatrix.elem(u1, u2)

    def weight(self, u1, u2):
        if u1 in self.followees and u2 in self.followees[u1]:
            return self.followees[u1][u2]
        else:
            return 0

    def trustSize(self):
        return self.trustMatrix.size

    def getFollowers(self, u):
        if u in self.followers:
            return self.followers[u]
        else:
            return {}

    def getFollowees(self, u):
        if u in self.followees:
            return self.followees[u]
        else:
            return {}

    def hasFollowee(self, u1, u2):
        if u1 in self.followees:
            if u2 in self.followees[u1]:
                return True
            else:
                return False
        return False

    def hasFollower(self, u1, u2):
        if u1 in self.followers:
            if u2 in self.followers[u1]:
                return True
            else:
                return False
        return False


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
        self.is_train = True if "train" in file_list or file_list else False
        self.trainingSet = loadDataSet(
            config.get("runner.rating_file", None),
            bTest=False,
            binarized=True,
            threshold=1.0)
        self.relation = loadRelationship(
            config.get("runner.relation_file", None))
        self.social = Social(relation=self.relation)
        self.batch_size = config.get("runner.train_batch_size", 2000)

        for trainingSet, testSet in crossValidation(self.trainingSet, k=5):
            self.data = Rating(trainingSet, testSet)
            self.trainingSet = trainingSet
            self.testSet = testSet
            break

        _, _, self.train_size = self.data.trainingSize()
        _, _, self.test_size = self.data.testSize()

    def get_dataset(self):
        # data clean
        cleanList = []
        cleanPair = []
        for user in self.social.followees:
            if user not in self.data.user:
                cleanList.append(user)
            for u2 in self.social.followees[user]:
                if u2 not in self.data.user:
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.social.followees[u]
        for pair in cleanPair:
            if pair[0] in self.social.followees:
                del self.social.followees[pair[0]][pair[1]]
        cleanList = []
        cleanPair = []
        for user in self.social.followers:
            if user not in self.data.user:
                cleanList.append(user)
            for u2 in self.social.followers[user]:
                if u2 not in self.data.user:
                    cleanPair.append((user, u2))
        for u in cleanList:
            del self.social.followers[u]
        for pair in cleanPair:
            if pair[0] in self.social.followers:
                del self.social.followers[pair[0]][pair[1]]
        idx = []
        for n, pair in enumerate(self.social.relation):
            if pair[0] not in self.data.user or pair[1] not in self.data.user:
                idx.append(n)
        for item in reversed(idx):
            del self.social.relation[item]
        return self.data, self.social

    def __iter__(self):
        count = 0
        item_list = list(self.data.item.keys())

        if self.is_train:
            shuffle(self.data.trainingData)
            while count < self.train_size:
                output_list = []

                user, item = self.data.trainingData[count][
                    0], self.data.trainingData[count][1]
                neg_item = choice(item_list)
                while neg_item in self.data.trainSet_u[user]:
                    neg_item = choice(item_list)

                output_list.append(
                    np.array(self.data.user[user]).astype("int64"))
                output_list.append(
                    np.array(self.data.item[item]).astype("int64"))
                output_list.append(
                    np.array(self.data.item[neg_item]).astype("int64"))

                count += 1
                yield output_list
        else:
            while count < self.test_size:
                output_list = []

                user, item = self.data.testData[count][0], self.data.testData[
                    count][1]
                neg_item = choice(item_list)

                output_list.append(
                    np.array(self.data.user[user]).astype("int64"))
                output_list.append(
                    np.array(self.data.item[item]).astype("int64"))
                output_list.append(
                    np.array(self.data.item[neg_item]).astype("int64"))

                count += 1
                yield output_list
