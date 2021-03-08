# -*- coding=utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import codecs
import os
import time
import collections
import argparse
import warnings
import multiprocessing as mp
import numpy as np

from sklearn.cluster import KMeans
import tree_builder

__all__ = ['Cluster']


class Cluster:
    def __init__(self,
                 filename,
                 emb_size,
                 parall=16,
                 prev_result=None,
                 output_dir='./',
                 _n_clusters=2):
        self.emb_size = emb_size
        self.mini_batch = 256
        self.parall = parall
        self.filename = filename
        self.output_dir = output_dir
        self.n_clusters = _n_clusters
        self.timeout = 5

        self.ids = None
        self.data = None
        self.items = None
        self.queue = None
        self.codes = None
        self.count = 0
        self.prev_result = prev_result

    def _read(self):
        start_time = time.time()
        ids = list()
        data = list()
        items = dict()
        with codecs.open(self.filename, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                self._line_process(line, ids, data, items)
                if index % 1000 == 0:
                    cur_time = time.time()
                    print("Read {} line, speed: {} line/s".format(
                        index, (cur_time - start_time) / (index+1)))

        for i in range(len(data)):
            data_len = len(data[0])
            if data_len != 1:
                for j in range(1, data_len):
                    data[i][j] /= data[i][0]
            data[i] = data[i][1:]

        self.ids = np.array(ids)
        self.data = np.array(data)
        self.items = items
        end_time = time.time()
        print("Read data done, {} records read, using time: {}".format(
            len(ids), end_time - start_time))

    def _line_process(self, line, ids, data, items):
        arr = line.rstrip().split('\t')

        if not arr:
            warnings.warn("Illegal input encountered. Error type: empty line")
            return
        elif len(arr) == 1:
            label = arr[0]
            emb_vec = (np.random.random_sample(
                (self.emb_size))).tolist()
        elif len(arr) == 2:
            label = arr[1]
            emb_vec = arr[0].split()

        if len(emb_vec) != self.emb_size:
            warnings.warn("Illegal input encountered. Error type: embedding vec length {} doesn't meet requirment {}".format(
                len(emb_vec), self.emb_size))
            return

        if label not in items:
            items[label] = self.count
            ids.append(self.count)
            self.count += 1
            # 保存item的vec, 首位记录遇到的次数
            vector = list()
            vector.append(1)
            for i in range(0, len(emb_vec)):
                vector.append(float(emb_vec[i]))
            data.append(vector)
        else:
            # 相同item遇到多次, 记录每次的emb值，之后取均值
            index = items[label]
            for i in range(0, len(emb_vec)):
                data[index][i + 1] += float(emb_vec[i])
            data[index][0] += 1

    def train(self):
        ''' multi process cluster '''
        self._read()

        queue = mp.Queue()
        self.process_prev_result(queue)

        processes = []
        pipes = []
        for _ in range(self.parall):
            a, b = mp.Pipe()
            p = mp.Process(target=self._train, args=(b, queue))
            processes.append(p)
            pipes.append(a)
            p.start()

        self.codes = np.zeros((len(self.ids), ), dtype=np.int64)
        for pipe in pipes:
            codes = pipe.recv()
            for i in range(len(codes)):
                if codes[i] > 0:
                    self.codes[i] = codes[i]

        for p in processes:
            p.join()

        assert (queue.empty())
        builder = tree_builder.TreeBuilder(self.output_dir, self.n_clusters)
        builder.build(self.ids, self.codes,
                      items=self.items, data=self.data)

    def process_prev_result(self, queue):
        if not self.prev_result:
            queue.put((0, np.array(range(len(self.ids)))))
            return True

        di = dict()
        for i, node_id in enumerate(self.ids):
            di[node_id] = i

        indexes = []
        clusters = []
        with open(self.prev_result) as f:
            for line in f:
                arr = line.split(",")
                if arr < 2:
                    break
                ni = [di[int(m)] for m in arr]
                clusters.append(ni)
                indexes += ni
        assert len(set(indexes)) == len(self.ids), \
            "ids count: {}, index count: {}".format(len(self.ids),
                                                    len(set(indexes)))
        count = len(clusters)
        assert (count & (count - 1)) == 0, \
            "Prev cluster count: {}".format(count)
        for i, ni in enumerate(clusters):
            queue.put((i + count - 1, np.array(ni)))
        return True

    def _train(self, pipe, queue):
        last_size = -1
        catch_time = 0
        processed = False
        code = np.zeros((len(self.ids), ), dtype=np.int64)
        while True:
            for _ in range(3):
                try:
                    pcode, index = queue.get(timeout=self.timeout)
                except:
                    index = None
                if index is not None:
                    break

            if index is None:
                if processed and (last_size <= self.mini_batch or
                                  catch_time >= 3):
                    print("Process {} exits".format(os.getpid()))
                    break
                else:
                    print("Got empty job, pid: {}, time: {}".format(os.getpid(
                    ), catch_time))
                    catch_time += 1
                    continue

            processed = True
            catch_time = 0
            last_size = len(index)
            if last_size <= self.mini_batch:
                self._minbatch(pcode, index, code)
            else:
                start = time.time()
                sub_index = self._cluster(index)
                if last_size > self.mini_batch:
                    print("Train iteration done, pcode:{}, "
                          "data size: {}, using time: {}"
                          .format(pcode, len(index), time.time() - start))
                self.timeout = int(0.4 * self.timeout + 0.6 * (time.time() -
                                                               start))
                if self.timeout < 5:
                    self.timeout = 5

                for i in range(self.n_clusters):
                    if len(sub_index[i]) > 1:
                        queue.put(
                            (self.n_clusters * pcode + i + 1, sub_index[i]))

        process_count = 0
        for c in code:
            if c > 0:
                process_count += 1
        print("Process {} process {} items".format(os.getpid(), process_count))
        pipe.send(code)

    def _minbatch(self, pcode, index, code):
        dq = collections.deque()
        dq.append((pcode, index))
        batch_size = len(index)
        tstart = time.time()
        while dq:
            pcode, index = dq.popleft()
            if len(index) <= self.n_clusters:
                for i in range(len(index)):
                    code[index[i]] = self.n_clusters * pcode + i + 1
                continue

            sub_index = self._cluster(index)
            for i in range(self.n_clusters):
                if len(sub_index[i]) > 1:
                    dq.append((self.n_clusters * pcode + i + 1, sub_index[i]))
                elif len(sub_index[i]) > 0:
                    for j in range(len(sub_index[i])):
                        code[sub_index[i][j]] = self.n_clusters * \
                            pcode + i + j + 1
        print("K-means minbatch, batch size: {}, using time: {}".format(
            batch_size, time.time() - tstart))

    def _cluster(self, index):
        data = self.data[index]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(data)
        labels = kmeans.labels_
        sub_indexes = []
        remain_index = []
        ave_num = int(len(index) / self.n_clusters)

        # do Re-balance
        for i in range(self.n_clusters):

            sub_i = np.where(labels == i)[0]
            sub_index = index[sub_i]
            if len(sub_index) <= ave_num:
                sub_indexes.append(sub_index)
            else:
                distances = kmeans.transform(data[sub_i])[:, i]
                sorted_index = sub_index[np.argsort(distances)]
                sub_indexes.append(sorted_index[:ave_num])
                remain_index.extend(list(sorted_index[ave_num:]))

        idx = 0
        while idx < self.n_clusters and len(remain_index) > 0:
            if len(sub_indexes[idx]) >= ave_num:
                idx += 1
            else:
                diff = min(len(remain_index), ave_num - len(sub_indexes[idx]))
                sub_indexes[idx] = np.append(sub_indexes[idx],
                                             np.array(remain_index[0:diff]))
                remain_index = remain_index[diff:]
                idx += 1
        if len(remain_index) > 0:
            sub_indexes[0] = np.append(sub_indexes[0], np.array(remain_index))

        return sub_indexes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tree cluster")
    parser.add_argument(
        "--embed_file",
        required=True,
        help="filename of the embedded vector file")
    parser.add_argument(
        "--emb_size",
        type=int,
        default=64,
        help="dimension of input embedded vector")
    parser.add_argument(
        "--parall",
        type=int,
        default=16,
        help="Parall execution process number")
    parser.add_argument(
        "--prev_result",
        default=None,
        help="filename of the previous cluster reuslt")
    parser.add_argument(
        "--n_cluster",
        type=int,
        default=2,
        help="Branches of trees")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Branches of trees")

    argments = parser.parse_args()
    t1 = time.time()
    cluster = Cluster(argments.embed_file, argments.emb_size, argments.parall,
                      argments.prev_result, output_dir=parser.output_dir, _n_clusters=parser.n_clusters)
    cluster.train()
    t2 = time.time()
    print("K-means cluster complete successfully, using time: {}".format(t2 - t1))
