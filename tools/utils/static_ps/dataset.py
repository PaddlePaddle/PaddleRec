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
""" Dataset Definition """

import os
import time
import threading
import numpy as np

import paddle
from paddle.distributed import fleet
from pgl.utils.logger import log
import traceback

import util
from place import get_cuda_places
import model_util as model_util
import queue
import helper

def compute_max_nodes(emb_size, allocate_rate):
    """compute the max unique nodes"""
    total_gpu_memory_bytes = paddle.device.cuda.get_device_properties().total_memory
    allocate_memory_bytes = total_gpu_memory_bytes * allocate_rate
    max_unique_nodes = int(allocate_memory_bytes / (emb_size * 4 * 2))
    log.info("max_gpu_memory: %s allocate_rate: %s max_uniq_nodes: %s" %
        (total_gpu_memory_bytes // (1024 ** 3), allocate_rate, max_unique_nodes))

    return max_unique_nodes

class BaseDataset(object):
    """ BaseDataset for PGLBox.
    """

    def __init__(self,
                 chunk_num,
                 config,
                 holder_list,
                 embedding=None,
                 dist_graph=None,
                 is_predict=False):
        self.exception_queue = queue.Queue()
        self.ins_ready_sem = threading.Semaphore(0)
        self.enable_multi_node_sampling = os.getenv("FLAGS_enable_graph_multi_node_sampling", "false")
        log.info("enable_multi_node_sampling[%s]", str(self.enable_multi_node_sampling))

        if self.enable_multi_node_sampling == "true":
            log.info("enable_multi_node_sampling is true")
            self.could_load_sem = threading.Semaphore(1)
        elif self.enable_multi_node_sampling == "false":
            log.info("enable_multi_node_sampling is false")
            self.could_load_sem = threading.Semaphore(2)
        else:
            log.info("enable_multi_node_sampling[%s]", str(self.enable_multi_node_sampling))

        self.dist_graph = dist_graph
        self.config = config
        self.embedding = embedding
        self.chunk_num = chunk_num
        self.holder_list = holder_list
        self.is_predict = is_predict

    def compute_chunks_and_cap(self, config):
        """compute the chunks by gpu allocated rate"""
        sage_mode = config.sage_mode if config.sage_mode else False

        gpups_memory_allocated_rate = config.gpups_memory_allocated_rate if config.gpups_memory_allocated_rate else 0.25
        train_pass_cap = infer_pass_cap = compute_max_nodes(config.emb_size, gpups_memory_allocated_rate)

        train_chunk_nodes = int(config.walk_len * config.walk_times * config.batch_size)
        infer_chunk_nodes = int(config.infer_batch_size)
        uniq_factor = 0.4

        if sage_mode:
            etype2files = helper.parse_files(config.etype2files)
            etype_list = util.get_all_edge_type(etype2files, config.symmetry)
            etype_len = len(etype_list)

            train_chunk_nodes *= np.prod(config.samples) * config.win_size
            infer_chunk_nodes *= np.prod(config.infer_samples)

        if config.train_pass_cap:
            train_pass_cap = config.train_pass_cap
        if config.infer_pass_cap:
            infer_pass_cap = config.infer_pass_cap

        train_sample_times_one_chunk = int(train_pass_cap / train_chunk_nodes / uniq_factor)
        infer_sample_times_one_chunk = int(infer_pass_cap / infer_chunk_nodes)

        train_sample_times_one_chunk = max(train_sample_times_one_chunk, 1)
        infer_sample_times_one_chunk = max(infer_sample_times_one_chunk, 1)

        log.info("sample_times_one_chunk: train [%s], infer [%s]" % \
                 (train_sample_times_one_chunk, infer_sample_times_one_chunk))
        ret =  {
            "train_pass_cap": train_pass_cap * 2,
            "infer_pass_cap": infer_pass_cap * 2 if self.enable_multi_node_sampling == "true" else infer_pass_cap,
            "train_sample_times_one_chunk": train_sample_times_one_chunk,
            "infer_sample_times_one_chunk": infer_sample_times_one_chunk,
        }
        return ret

    def generate_dataset(self, config, chunk_index, pass_num):
        """ generate dataset """
        sage_mode = config.sage_mode if config.sage_mode else False
        fs_name = config.fs_name if config.fs_name is not None else ""
        fs_ugi = config.fs_ugi if config.fs_ugi is not None else ""

        str_samples = util.sample_list_to_str(sage_mode, config.samples)
        str_infer_samples = util.sample_list_to_str(sage_mode,
                                                    config.infer_samples)

        excluded_train_pair = config.excluded_train_pair if config.excluded_train_pair else ""
        pair_label = config.pair_label if config.pair_label else ""
        infer_node_type = config.infer_node_type if config.infer_node_type else ""

        cap_and_chunks = self.compute_chunks_and_cap(config)

        get_degree = sage_mode and (config.use_degree_norm
                                    if config.use_degree_norm else False)
        
        is_sharding = False
        if getattr(config, "sharding", None):
            is_sharding = True

        if config.accumulate_num is None:
            accumulate_num = 1
        else:
            accumulate_num = config.accumulate_num

        if not config.model_type.startswith("Accum"):
            accumulate_num = 1

        graph_config = {
            "walk_len": config.walk_len,
            "walk_degree": config.walk_times,
            "once_sample_startid_len": config.batch_size,
            "sample_times_one_chunk": cap_and_chunks["train_sample_times_one_chunk"],
            "window": config.win_size,
            "debug_mode": config.debug_mode,
            "batch_size": config.batch_size,
            "meta_path": config.meta_path,
            "gpu_graph_training": not self.is_predict,
            "sage_mode": sage_mode,
            "samples": str_samples,
            "train_table_cap": cap_and_chunks["train_pass_cap"],
            "infer_table_cap": cap_and_chunks["infer_pass_cap"],
            "excluded_train_pair": excluded_train_pair,
            "infer_node_type": infer_node_type,
            "get_degree": get_degree,
            "weighted_sample": config.weighted_sample,
            "return_weight": config.return_weight,
            "pair_label": pair_label,
            "is_thread_sharding": is_sharding,
            "accumulate_num": accumulate_num,
        }

        graph_config["first_node_type"] = config.first_node_type

        if self.is_predict:
            graph_config["walk_len"] = 1
            graph_config["walk_degree"] = 1
            graph_config["batch_size"] = config.infer_batch_size
            graph_config["once_sample_startid_len"] = config.infer_batch_size
            graph_config["samples"] = str_infer_samples
            graph_config["sample_times_one_chunk"] = cap_and_chunks["infer_sample_times_one_chunk"]
            graph_config["accumulate_num"] = 1

        dataset = paddle.base.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_feed_type("SlotRecordInMemoryDataFeed")
        dataset.set_use_var(self.holder_list)
        dataset.set_graph_config(graph_config)

        dataset.set_batch_size(
            1)  # Fixed Don't Change. Batch Size is not config here.
        dataset.set_thread(len(get_cuda_places()))

        dataset.set_hdfs_config(fs_name, fs_ugi)

        dataset._set_use_ps_gpu(self.embedding.parameter_server)
        file_list = self.get_file_list(chunk_index)
        dataset.set_filelist(file_list)
        dataset.set_pass_id(pass_num)
        with open("datafeed.pbtxt", "w") as fout:
            fout.write(dataset.desc())
        return dataset

    def pass_generator(self):
        """ pass generator """
        raise NotImplementedError

    def get_file_list(self, chunk_index):
        """ get data file list """
        work_dir = "./workdir/filelist"  # a tmp directory that does not used in other places

        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        file_list = []
        for thread_id in range(len(get_cuda_places())):
            filename = os.path.join(work_dir, "%s_%s_%s" %
                                    (self.chunk_num, chunk_index, thread_id))
            file_list.append(filename)
            with open(filename, "w") as writer:
                writer.write("%s_%s_%s\n" %
                             (self.chunk_num, chunk_index, thread_id))
        return file_list

    def preload_thread(self, dataset_list, exception_queue):
        """ This is a thread to fill the dataset_list
        """
        try:
            self.preload_worker(dataset_list)
        except Exception as e:
            self.could_load_sem.release()
            self.ins_ready_sem.release()
            log.warning('preload_thread exception')
            exception_queue.put(e)

    def preload_worker(self, dataset_list):
        """ This is a preload worker to generate pass dataset asynchronously
        """

        global pass_id
        global epoch_stat
        pass_id = 0
        epoch_stat = 0
        while 1:
            dataset = None
            self.could_load_sem.acquire()
            if dataset is not None:
                dataset.wait_preload_done()

            if dataset is None:
                index = fleet.worker_index() * self.chunk_num
                global_chunk_num = fleet.worker_num() * self.chunk_num

                if dataset is None:
                    index = fleet.worker_index() * self.chunk_num
                    global_chunk_num = fleet.worker_num() * self.chunk_num

                epoch_finish = False
                dataset = self.generate_dataset(self.config, index, pass_id)
                begin = time.time()
                try:
                    dataset.load_into_memory(is_shuffle=False)
                except Exception as e:
                    log.warning('load_into_memory exception')
                    self.ins_ready_sem.release()
                    raise e
                    break
                end = time.time()
                log.info("pass[%d] STAGE [GPU SAMPLE] finished, time cost: %f sec",
                         pass_id, end - begin)
                if self.config.need_dump_walk is True and self.is_predict is False:
                    dataset.dump_walk_path(self.config.local_dump_walk_path)
                if self.config.need_dump_neighbors is True and self.is_predict is True:
                    dataset.dump_sample_neighbors(self.config.local_dump_neighbors_path)
                dataset_list.append(dataset)
                pass_id = pass_id + 1
                if not self.is_predict:
                    epoch_finish = dataset.get_epoch_finish()
                    if epoch_finish:
                        if paddle.distributed.get_world_size() > 1:
                            epoch_stat = 1
                            log.info("get epoch finish in multi node, set epoch stat")
                self.ins_ready_sem.release()

                # Only training has epoch finish == True.
                if not self.is_predict:
                    if paddle.distributed.get_world_size() > 1:
                        multi_node_stat = util.allreduce_min(epoch_stat)
                        if multi_node_stat == 1:
                            log.info("all epoch stat is 1, will clear state and exit")
                            dataset = self.generate_dataset(self.config, index, pass_id)
                            dataset.clear_sample_state()
                            self.ins_ready_sem.release()
                            break
                        else:
                            if epoch_stat == 1:
                                log.info("other node has pass, not break")
                        util.barrier()

                    if epoch_finish:
                        if paddle.distributed.get_world_size() == 1:
                            log.info("epoch_finish == true, break")
                            self.ins_ready_sem.release()
                            break

                    if self.config.max_steps > 0 and model_util.print_count >= self.config.max_steps:
                        log.info(
                            "reach max_steps, dataset generator break")
                        self.ins_ready_sem.release()
                        break

                else:
                    data_size = dataset.get_memory_data_size()
                    if data_size == 0:
                        log.info("infer memory data_size == 0, break")
                        self.ins_ready_sem.release()
                        break

class UnsupReprLearningDataset(BaseDataset):
    """Unsupervised representation learning dataset.
    """

    def __init__(self,
                 chunk_num,
                 dataset_config,
                 holder_list,
                 embedding=None,
                 dist_graph=None):

        self.dataset_config = dataset_config

        super(UnsupReprLearningDataset, self).__init__(
            chunk_num=chunk_num,
            config=dataset_config,
            holder_list=holder_list,
            embedding=embedding,
            dist_graph=dist_graph,
            is_predict=False)

    def pass_generator(self, epoch=None):
        """ pass generator, open a thread for processing the data
        """
        dataset_list = []
        t = threading.Thread(target=self.preload_thread, args=(dataset_list, self.exception_queue))
        t.setDaemon(True)
        t.start()

        pass_id = 0
        while 1:
            self.ins_ready_sem.acquire()

            if len(dataset_list) == 0:
                log.info("train pass[%d] dataset_list is empty" % (pass_id))
                break

            dataset = dataset_list.pop(0)
            if dataset is None:
                log.info("train pass[%d] dataset is null" % (pass_id))
                self.could_load_sem.release()
                continue

            data_size = dataset.get_memory_data_size()
            if data_size == 0:
                log.info("train pass[%d], dataset size is 0" % (pass_id))
                self.could_load_sem.release()
                continue

            if self.config.max_steps > 0 and model_util.print_count >= self.config.max_steps:
                log.info("reach max_steps: %d, epoch[%d] train end" %
                         (self.config.max_steps, epoch))
                self.embedding.begin_pass()
                dataset.release_memory()
                self.embedding.end_pass()
                self.could_load_sem.release()
                continue

            beginpass_begin = time.time()
            self.embedding.begin_pass()
            beginpass_end = time.time()
            log.info("train pass[%d] STAGE [GPU BEGIN PASS] finished, time cost: %f sec" \
                    % (pass_id, beginpass_end - beginpass_begin))
            trainpass_begin = time.time()

            yield dataset

            trainpass_end = time.time()
            log.info("train pass[%d] STAGE [GPU TRAIN] finished, time cost: %f sec" \
                    % (pass_id, trainpass_end - trainpass_begin))

            dataset.release_memory()
            endpass_begin = time.time()
            self.embedding.end_pass()
            endpass_end = time.time()
            log.info("train pass[%d] STAGE [GPU END PASS] finished, time cost: %f sec" \
                    % (pass_id, endpass_end - endpass_begin))
            self.could_load_sem.release()

            if pass_id % self.config.save_cache_frequency == 0:
                cache_pass_id = pass_id - self.config.mem_cache_passid_num
                cache_pass_id = 0 if cache_pass_id < 0 else cache_pass_id
                cache_begin = time.time()
                fleet.save_cache_table(0, cache_pass_id)
                cache_end = time.time()
                log.info(
                    "train pass[%d] STAGE [SSD CACHE TABLE] finished, time cost: %f sec",
                    pass_id, cache_end - cache_begin)

            pass_id = pass_id + 1

        t.join()
        if not self.exception_queue.empty():
            raise self.exception_queue.get()


class InferDataset(BaseDataset):
    """Infer dataset for graph embedding learning.
    """

    def __init__(self,
                 chunk_num,
                 dataset_config,
                 holder_list,
                 infer_model_dict,
                 embedding=None,
                 dist_graph=None):

        self.dataset_config = dataset_config
        self.infer_model_dict = infer_model_dict

        super(InferDataset, self).__init__(
            chunk_num=chunk_num,
            config=dataset_config,
            holder_list=holder_list,
            embedding=embedding,
            dist_graph=dist_graph,
            is_predict=True)

    def pass_generator(self):
        """ pass generator, open a thread for processing the data
        """
        dataset_list = []
        t = threading.Thread(target=self.preload_thread, args=(dataset_list, self.exception_queue))
        t.setDaemon(True)
        t.start()

        pass_id = 0
        while 1:
            self.ins_ready_sem.acquire()

            if len(dataset_list) == 0:
                log.info("infer pass[%d] dataset_list is empty" % (pass_id))
                break

            dataset = dataset_list.pop(0)
            if dataset is None:
                log.info("infer pass[%d] dataset is null" % (pass_id))
                self.could_load_sem.release()
                continue

            data_size = dataset.get_memory_data_size()
            if data_size == 0:
                log.info("infer pass[%d] dataset size is 0" % (pass_id))
                self.could_load_sem.release()
                continue

            beginpass_begin = time.time()
            self.embedding.begin_pass()
            beginpass_end = time.time()
            log.info(
                "infer pass[%d] STAGE [BEGIN PASS] finished, time cost: %f sec",
                pass_id, beginpass_end - beginpass_begin)
            trainpass_begin = time.time()

            yield dataset

            trainpass_end = time.time()
            log.info("infer pass[%d] STAGE [TRAIN] finished, time cost: %f sec" \
                    % (pass_id, trainpass_end - trainpass_begin))

            dataset.release_memory()
            endpass_begin = time.time()
            self.embedding.end_pass()
            endpass_end = time.time()
            log.info(
                "infer pass[%d] STAGE [END PASS] finished, time cost: %f sec",
                pass_id, endpass_end - endpass_begin)
            pass_id = pass_id + 1
            self.could_load_sem.release()
        t.join()
        if not self.exception_queue.empty():
            raise self.exception_queue.get()
