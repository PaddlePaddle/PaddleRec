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

from paddle.distributed.fleet.proto import index_dataset_pb2
import numpy as np
import struct
import argparse
import os
import time
import collections
import multiprocessing as mp

from sklearn.cluster import KMeans


class TreeIndexBuilder:
    def __init__(self):
        self.branch = 2
        self.timeout = 5
        self.id_emb = {}
        self.code_id = {}

    def build_by_category(self, input_filename, output_filename):
        class Item:
            def __init__(self, item_id, cat_id):
                self.item_id = item_id
                self.cat_id = cat_id
                self.code = 0

            def __lt__(self, other):
                return self.cat_id < other.cat_id or \
                    (self.cat_id == other.cat_id and
                     self.item_id < other.item_id)

        items = []
        item_id_set = set()
        with open(input_filename, 'r') as f:
            for line in f:
                iterobj = line.split()
                item_id = int(iterobj[0])
                cat_id = int(iterobj[1])
                if item_id not in item_id_set:
                    items.append(Item(item_id, cat_id))
                    item_id_set.add(item_id)
        del item_id_set
        items.sort()

        def gen_code(start, end, code):
            if end <= start:
                return
            if end == start + 1:
                items[start].code = code
                return
            num = int((end - start) / self.branch)
            remain = int((end - start) % self.branch)
            for i in range(self.branch):
                _sub_end = start + (i + 1) * num
                if (remain > 0):
                    remain -= 1
                    _sub_end += 1
                _sub_end = min(_sub_end, end)
                gen_code(start, _sub_end, self.branch * code + self.branch - i)
                start = _sub_end

        gen_code(0, len(items), 0)
        ids = np.array([item.item_id for item in items])
        codes = np.array([item.code for item in items])
        print('before build')
        self.build(output_filename, ids, codes)

    def get_id_code(self, _path):
        self.id_code = {}
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                key = line[1]
                self.id_code[key] = int(line[0])

    def codes_reader(self, path):
        codes = []
        with open(path, 'r') as f:
            for line in f:
                codes.append(line.strip())
        return codes

    def tree_init_by_kmeans(self, input_filename, output_filename, parall=1):
        t1 = time.time()
        ids = list()
        data = list()
        with open(input_filename) as f:
            for line in f:
                arr = line.split(',')
                if not arr:
                    break
                ids.append(int(arr[0]))
                vector = list()
                for i in range(1, len(arr)):
                    vector.append(float(arr[i]))
                data.append(vector)
        self.ids = np.array(ids)
        self.data = np.array(data)
        t2 = time.time()
        print("Read data done, {} records read, elapsed: {}".format(
            len(ids), t2 - t1))

        queue = mp.Queue()
        queue.put((0, np.array(range(len(self.ids)))))
        processes = []
        pipes = []
        for _ in range(parall):
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

        self.build(output_filename, self.ids, self.codes, data=self.data)

    def _train(self, pipe, queue):
        last_size = -1
        catch_time = 0
        processed = False
        code = np.zeros((len(self.ids), ), dtype=np.int64)
        while True:
            for _ in range(5):
                try:
                    pcode, index = queue.get(timeout=self.timeout)
                except:
                    index = None
                if index is not None:
                    break

            if index is None:
                if processed and (last_size <= 1024 or catch_time >= 3):
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
            if last_size <= 1024:
                self._minbatch(pcode, index, code)
            else:
                tstart = time.time()
                left_index, right_index = self._cluster(index)
                if last_size > 1024:
                    print("Train iteration done, pcode:{}, "
                          "data size: {}, elapsed time: {}"
                          .format(pcode, len(index), time.time() - tstart))
                self.timeout = int(0.4 * self.timeout + 0.6 * (time.time() -
                                                               tstart))
                if self.timeout < 5:
                    self.timeout = 5

                if len(left_index) > 1:
                    queue.put((2 * pcode + 1, left_index))

                if len(right_index) > 1:
                    queue.put((2 * pcode + 2, right_index))
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

            if len(index) == 2:
                code[index[0]] = 2 * pcode + 1
                code[index[1]] = 2 * pcode + 2
                continue

            left_index, right_index = self._cluster(index)
            if len(left_index) > 1:
                dq.append((2 * pcode + 1, left_index))
            elif len(left_index) == 1:
                code[left_index] = 2 * pcode + 1

            if len(right_index) > 1:
                dq.append((2 * pcode + 2, right_index))
            elif len(right_index) == 1:
                code[right_index] = 2 * pcode + 2

        print("Minbatch, batch size: {}, elapsed: {}".format(
            batch_size, time.time() - tstart))

    def _cluster(self, index):
        data = self.data[index]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_
        l_i = np.where(labels == 0)[0]
        r_i = np.where(labels == 1)[0]
        left_index = index[l_i]
        right_index = index[r_i]
        if len(right_index) - len(left_index) > 1:
            distances = kmeans.transform(data[r_i])
            left_index, right_index = self._rebalance(left_index, right_index,
                                                      distances[:, 1])
        elif len(left_index) - len(right_index) > 1:
            distances = kmeans.transform(data[l_i])
            left_index, right_index = self._rebalance(right_index, left_index,
                                                      distances[:, 0])

        return left_index, right_index

    def _rebalance(self, lindex, rindex, distances):
        sorted_index = rindex[np.argsort(distances)[::-1]]
        idx = np.concatenate((lindex, sorted_index))
        mid = int(len(idx) / 2)
        return idx[mid:], idx[:mid]

    def get_id_emb_bk(self, id, leaf_left, leaf_right):
        if self.id_emb.get(id) is not None:
            return self.id_emb[id]
        if id > leaf_right:
            return None
        if id >= leaf_left and id <= leaf_right and (id in self.codes):
            if self.id_emb.get(id) is not None:
                return self.id_emb[id]
            else:
                return None
        id_left_child = self.get_id_emb(2 * id + 1, leaf_left, leaf_right)
        id_right_child = self.get_id_emb(2 * id + 2, leaf_left, leaf_right)

        if id_left_child is not None and id_right_child is not None:
            tmp = []
            for i in range(len(id_left_child)):
                tmp.append(float(id_left_child[i] + id_right_child[i]) / 2)
            assert tmp[0] != ' ', 'tmp!!!'
            self.id_emb[id] = tmp
        elif id_left_child is not None:
            self.id_emb[id] = id_left_child
            if id == 8191:
                print('left!!!')
            assert len(id_left_child), 'left!!!'
        elif id_right_child is not None:
            if id == 8191:
                print('right!!!')
            self.id_emb[id] = id_right_child
            assert len(id_right_child), 'right!!!'
        else:
            return None
        if self.id_emb.get(id):
            assert len(self.id_emb[id]), 'return!!!'
            return self.id_emb[id]
        else:
            return None

    def assign_value(self, k):
        res = 0.0
        tmp = []
        if self.id_emb.get(2 * k + 1) is not None and self.id_emb.get(
                2 * k + 2) is not None:
            for i in range(len(self.id_emb[2 * k + 1])):
                tmp.append(
                    float(self.id_emb[2 * k + 1][i] + self.id_emb[2 * k + 2][
                        i]) / 2)
            self.id_emb[k] = tmp
        elif self.id_emb.get(2 * k + 1) is not None:
            self.id_emb[k] = self.id_emb[2 * k + 1]
        elif self.id_emb.get(2 * k + 2) is not None:
            self.id_emb[k] = self.id_emb[2 * k + 2]
        else:
            print('nonLeafNode should have real_child!!!!', k)
            exit(0)

    def get_id_emb(self, id, leaf_left, leaf_right):
        last_nonleaf_node = int((leaf_right - 1) / 2)
        node_point = last_nonleaf_node
        while node_point:
            self.assign_value(node_point)
            node_point -= 1
        assert node_point == 0, 'it have shoule been root node!!!'
        self.assign_value(node_point)

    def save_node_info(self, codes, data, ids, id_offset, leaf_left,
                       leaf_right):
        data = data.tolist()
        for code, vec in zip(codes, data):
            assert len(vec), 'vec!!!'
            self.id_emb[code] = vec

        print('leaf_left: ', leaf_left)
        print('leaf_right: ', leaf_right)

        print('befor dic len: ', len(self.id_emb))
        self.get_id_emb(0, leaf_left, leaf_right)
        len_dic = len(self.id_emb)
        print('dic len: ', len_dic)

        #tree_emb_np = np.zeros((len(self.id_emb), 128))
        tree_emb_ = []
        cnt = 0
        for i in range(leaf_right + 1):
            if self.id_emb.get(i) is not None:
                #tree_emb_np[cnt] = self.id_emb[i]
                tree_emb_.append(self.id_emb[i])
                self.code_id[i] = cnt
                cnt = cnt + 1
        tree_emb_np = np.array(tree_emb_).astype(np.float32)
        #tree_emb_np = np.zeros((leaf_right + 1, 128))
        #for key in self.id_emb.keys():
        #    tree_emb_np[key] = np.array(self.id_emb[key])
        np.save('./tree_emb.npy', tree_emb_np)

        with open('ids_id.txt', 'w') as f:
            for i in range(len(codes)):
                f.write(
                    str(ids[i]) + '\t' + str(self.code_id[codes[i]]) + '\n')
        f.close()

    def build(self, output_filename, ids, codes, data=None, id_offset=None):
        # process id offset
        '''
        if not id_offset:
            max_id = 0
            for id in ids:
                if id > max_id:
                    max_id = id
            id_offset = max_id + 1
        '''

        # sort by codes
        '''
        argindex = np.argsort(codes)
        codes = codes[argindex]
        ids = ids[argindex]
        data = data[argindex]
        '''

        # Trick, make all leaf nodes to be in same level
        min_code = 0
        max_code = codes[-1]
        for i in codes:
            max_code = max(i, max_code)
        while max_code > 0:
            min_code = min_code * self.branch + 1
            max_code = int((max_code - 1) / self.branch)

        for i in range(len(codes)):
            while codes[i] < min_code:
                codes[i] = codes[i] * self.branch + 1
        leaf_left = codes[0]
        leaf_right = codes[0]
        for i in codes:
            leaf_left = min(i, leaf_left)
            leaf_right = max(i, leaf_right)

        if not id_offset:
            id_offset = leaf_right - leaf_left + 1

        self.save_node_info(codes, data, ids, id_offset, leaf_left, leaf_right)
        filter_set = set()
        max_level = 0
        tree_meta = index_dataset_pb2.TreeMeta()
        print('reading...')
        with open(output_filename, 'wb') as f:
            for num, (item_name, code) in enumerate(zip(ids, codes)):
                node = index_dataset_pb2.IndexNode()
                node.id = self.code_id[code]
                node.item_name = str(item_name)
                node.is_leaf = True
                node.probability = 1.0

                kv_item = index_dataset_pb2.KVItem()
                kv_item.key = self._make_key(code)
                kv_item.value = node.SerializeToString()
                self._write_kv(f, kv_item.SerializeToString())

                ancessors = self._ancessors(code)
                if len(ancessors) + 1 > max_level:
                    max_level = len(ancessors) + 1

                for ancessor in ancessors:
                    if ancessor not in filter_set:
                        node = index_dataset_pb2.IndexNode()
                        node.id = ancessor  # id = id_offset + code
                        node.is_leaf = False
                        node.probability = 1.0
                        kv_item = index_dataset_pb2.KVItem()
                        kv_item.key = self._make_key(ancessor)
                        kv_item.value = node.SerializeToString()
                        self._write_kv(f, kv_item.SerializeToString())
                        filter_set.add(ancessor)

            tree_meta.branch = self.branch
            tree_meta.height = max_level
            kv_item = index_dataset_pb2.KVItem()
            kv_item.key = '.tree_meta'.encode('utf-8')

            kv_item.value = tree_meta.SerializeToString()
            self._write_kv(f, kv_item.SerializeToString())

    def _ancessors(self, code):
        ancs = []
        while code > 0:
            code = int((code - 1) / self.branch)
            ancs.append(code)
        return ancs

    def _make_key(self, code):
        return str(code).encode('utf-8')

    def _write_kv(self, fwr, message):
        fwr.write(struct.pack('i', len(message)))
        fwr.write(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TreeIndexBuiler")
    parser.add_argument(
        "--parallel",
        required=False,
        type=int,
        default=12,
        help="parallel nums.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=['by_category', 'by_kmeans'],
        help="mode")
    parser.add_argument("--input", required=True, help="input filename")
    parser.add_argument("--output", required=True, help="output filename")

    args = parser.parse_args()
    t1 = time.time()
    if args.mode == "by_category":
        builder = TreeIndexBuilder()
        builder.build_by_category(args.input, args.output)
    elif args.mode == "by_kmeans":
        builder = TreeIndexBuilder()
        builder.tree_init_by_kmeans(args.input, args.output, args.parallel)
    t2 = time.time()
    print('DONE!')
    print('spend {} seconds!'.format(t2 - t1))
