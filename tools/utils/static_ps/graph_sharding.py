# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Graph Sharding for PGLBox
"""

import sys
import os
import logging
import time
import json
import shutil
import argparse
import multiprocessing
from multiprocessing import Lock
from multiprocessing import Process
from multiprocessing import Queue

LOG_FILE = "shard.log"
logger = logging.getLogger('graph_sharding')
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(levelname)s: %(asctime)s %(process)d'
                        ' [%(filename)s:%(lineno)s][%(funcName)s] %(message)s')
debug_handler = logging.StreamHandler()
#  debug_handler = logging.FileHandler(LOG_FILE, 'a')
debug_handler.setFormatter(fmt)
debug_handler.setLevel(logging.DEBUG)
logger.addHandler(debug_handler)
#  print("the sharding log will be saved in %s" % LOG_FILE)

TEMP_FILE = ".tmp_sharding_file"
CPU_NUM = multiprocessing.cpu_count()


def mapper(args, input_file, lock_list, proc_index):
    """
    mapper
    """
    logger.debug('processing %s' % input_file)
    start_time = time.time()
    cache = []
    for i in range(args.num_part):
        cache.append([])

    if args.mock_float_feature:
      float_str = ",".join(["0.5", "0.5", "0.5", "0.5", "0.5"])
      float_slot = "9109:" + float_str
      float_str_2 = ",".join(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
      float_slot_2 = "9110:" + float_str_2
    line_cnt = 0
    with open(input_file, "r") as reader:
        for line in reader:
            try:
                line_cnt = line_cnt + 1
                line = line.rstrip("\r\n")
                fields = line.split("\t")
                if fields[0].isdigit(): # if is digit, it is edge file
                    src_nid = int(fields[0])
                    cache[src_nid % args.num_part].append(line)

                    if args.symmetry:
                        dst_nid = int(fields[1])
                        if src_nid != dst_nid:
                            cache[dst_nid % args.num_part].append(line)
          
                else:  # node shard
                    # append float feature
                    if args.mock_float_feature:
                      new_line = line
                      if line_cnt % 5 != 0:
                        new_line = new_line + "\t" + float_slot
                      if line_cnt % 10 != 0:
                        new_line = new_line + "\t" + float_slot_2
                    nid = int(fields[1])
                    if args.mock_float_feature:
                      cache[nid % args.num_part].append(new_line)
                    else:
                      cache[nid % args.num_part].append(line)
            except Exception as e:
                logger.debug(["[SHARDING_rank_%s] error msg: " % proc_index, e])
                logger.debug(["[SHARDING_rank_%s] error line: " % proc_index, line])

    read_time = time.time()

    for i in range(proc_index, args.num_part + proc_index):
        index = i % args.num_part
        lock_list[index].acquire()
        with open(os.path.join(args.output_dir, "part-%05d" % index),
                  'a+') as writer:
            for item in cache[index]:
                writer.write("%s\n" % item)
        lock_list[index].release()

    output_time = time.time()
    logger.debug('processed %s, read_time[%f] write_time[%f]' % (
        input_file, (read_time - start_time), (output_time - read_time)))


class FileMapper(object):
    """
    FileMapper
    """

    def init(self, input_dir):
        """ Init """
        self.exception_queue = Queue()
        if os.path.isdir(input_dir):
            file_list = os.listdir(input_dir)
            new_file_list = []
            for file in file_list:
                new_file_list.append(input_dir + '/' + file)
            file_list = new_file_list
        else:
            file_list = [input_dir]

        with open(TEMP_FILE, 'w') as f:
            json.dump(file_list, f)

        return len(file_list)

    def fini(self):
        """ Fini """
        os.system('rm -rf %s' % TEMP_FILE)
        sys.stdout.flush()

    def excute_func(self, args, func, progress_file_lock, lock_list,
                    proc_index, exception_queue):
        """ Repeat download one file """
        try:
            while (1):
                progress_file_lock.acquire()
                file_list = []
            
                with open(TEMP_FILE) as f:
                    file_list = json.load(f)
                if (len(file_list) == 0):
                    progress_file_lock.release()
                    break
            
                input_file = file_list[0]
                del file_list[0]
                with open(TEMP_FILE, 'w') as f:
                    json.dump(file_list, f)
                progress_file_lock.release()

                func(args, input_file, lock_list, proc_index)
        except Exception as e:
            logger.warning('preload_thread exception')
            exception_queue.put(e)

    def run(self, args, func, input_path):
        """ Run """
        file_num = self.init(input_path)
        process_list = []
        lock_list = []
        progress_file_lock = Lock()
        for i in range(args.num_part):
            lock_list.append(Lock())
        process_num = min(args.max_workers, file_num, CPU_NUM)
        for i in range(0, process_num):
            p = Process(
                target=self.excute_func,
                args=(args, func, progress_file_lock, lock_list, i, self.exception_queue))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
        if not self.exception_queue.empty():
            raise self.exception_queue.get()
        self.fini()

def makedirs(path):
    """ doc """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='graph_sharding')
    parser.add_argument("--input_dir", type=str, default="/pglbox/raw_data")
    parser.add_argument("--input_sub_dir", type=str, default="")
    parser.add_argument("--symmetry", action='store_true')
    parser.add_argument("--num_part", type=int, default=1000)
    parser.add_argument("--max_workers", type=int, default=200)
    parser.add_argument("--mock_float_feature", action='store_true')
    args = parser.parse_args()
    logger.debug(args)

    file_mapper = FileMapper()
    
    start = time.time()
    logger.debug("Graph Sharding of %s" % args.input_dir)
    if (len(args.input_sub_dir) == 0):
        file_or_dir_list = os.listdir(args.input_dir)
    else:
        file_or_dir_list = list(set(args.input_sub_dir.split(',')))
    logger.debug("file_or_dir_list: %s" % str(file_or_dir_list))

    for file_or_dir in file_or_dir_list:
        if len(file_or_dir) == 0:
            continue
        input_path = os.path.join(args.input_dir, file_or_dir)
        output_path = os.path.join(args.input_dir, "%s_tmp" % file_or_dir)
        makedirs(output_path)
        args.output_dir = output_path
        logger.debug("sharding from [%s] to [%s]" % (input_path, output_path))
        file_mapper.run(args, mapper, input_path)
        logger.debug("sharded from [%s] to [%s]" % (input_path, output_path))

        # remove origin data
        os.system("rm -rf %s" % input_path)
        os.system('mv %s %s' % (output_path, input_path))
    end = time.time()
    logger.debug("sharding time: %.2f" % (end - start))
    logger.debug("graph data sharding finished")
