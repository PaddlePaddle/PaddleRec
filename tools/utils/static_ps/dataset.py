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

import paddle
from paddle.distributed import fleet
from pgl.utils.logger import log

import util
from place import get_cuda_places
import model_util as model_util


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
        self.ins_ready_sem = threading.Semaphore(0)
        # multiple machines need to shut down the pipe, otherwise it will nccl block
        if paddle.distributed.get_world_size() > 1:
            self.could_load_sem = threading.Semaphore(1)
        else:
            self.could_load_sem = threading.Semaphore(2)
        self.dist_graph = dist_graph
        self.config = config
        self.embedding = embedding
        self.chunk_num = chunk_num
        self.holder_list = holder_list
        self.is_predict = is_predict

    def generate_dataset(self, config, chunk_index, pass_num):
        """ generate dataset """
        sage_mode = config.sage_mode if config.sage_mode else False
        fs_name = config.fs_name if config.fs_name is not None else ""
        fs_ugi = config.fs_ugi if config.fs_ugi is not None else ""

        str_samples = util.sample_list_to_str(sage_mode, config.samples)
        str_infer_samples = util.sample_list_to_str(sage_mode,
                                                    config.infer_samples)

        excluded_train_pair = config.excluded_train_pair if config.excluded_train_pair else ""
        infer_node_type = config.infer_node_type if config.infer_node_type else ""

        uniq_factor = 0.4
        if not sage_mode:
            train_pass_cap = int(config.walk_len * config.walk_times * config.sample_times_one_chunk \
                             * config.batch_size * uniq_factor)
        else:
            # If sage_mode is True, self.samples can not be None.
            train_pass_cap = int(config.walk_len * config.walk_times * config.sample_times_one_chunk \
                             * config.batch_size * uniq_factor * config.samples[0])

        infer_pass_cap = 10000000  # 1kw
        if config.train_pass_cap:
            train_pass_cap = config.train_pass_cap
        if config.infer_pass_cap:
            infer_pass_cap = config.infer_pass_cap

        get_degree = sage_mode and (config.use_degree_norm
                                    if config.use_degree_norm else False)

        graph_config = {
            "walk_len": config.walk_len,
            "walk_degree": config.walk_times,
            "once_sample_startid_len": config.batch_size,
            "sample_times_one_chunk": config.sample_times_one_chunk,
            "window": config.win_size,
            "debug_mode": config.debug_mode,
            "batch_size": config.batch_size,
            "meta_path": config.meta_path,
            "gpu_graph_training": not self.is_predict,
            "sage_mode": sage_mode,
            "samples": str_samples,
            "train_table_cap": train_pass_cap,
            "infer_table_cap": infer_pass_cap,
            "excluded_train_pair": excluded_train_pair,
            "infer_node_type": infer_node_type,
            "get_degree": get_degree
        }

        first_node_type = util.get_first_node_type(config.meta_path)
        graph_config["first_node_type"] = first_node_type

        if self.is_predict:
            graph_config["batch_size"] = config.infer_batch_size
            graph_config["samples"] = str_infer_samples

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

    def preload_thread(self, dataset_list):
        """ This is a thread to fill the dataset_list
        """
        try:
            self.preload_worker(dataset_list)
        except Exception as e:
            self.could_load_sem.release()
            self.ins_ready_sem.release()
            log.warning('preload_thread exception :%s' % (e))

    def preload_worker(self, dataset_list):
        """ This is a preload worker to generate pass dataset asynchronously
        """

        global pass_id
        pass_id = 0
        while 1:
            dataset = None
            self.could_load_sem.acquire()
            if dataset is not None:
                dataset.wait_preload_done()

            if dataset is None:
                index = fleet.worker_index() * self.chunk_num
                global_chunk_num = fleet.worker_num() * self.chunk_num

                dataset = self.generate_dataset(self.config, index, pass_id)
                begin = time.time()
                dataset.load_into_memory(is_shuffle=False)
                end = time.time()
                log.info("pass[%d] STAGE [SAMPLE] finished, time cost: %f sec",
                         pass_id, end - begin)

                dataset_list.append(dataset)
                pass_id = pass_id + 1
                self.ins_ready_sem.release()

                # Only training has epoch finish == True.
                if not self.is_predict:

                    if self.config.metapath_split_opt:
                        data_size = dataset.get_memory_data_size()
                        if data_size == 0:
                            log.info(
                                "train metapath memory data_size == 0, break")
                            self.ins_ready_sem.release()
                            break
                    else:
                        data_size = dataset.get_memory_data_size()
                        epoch_finish = dataset.get_epoch_finish()
                        if epoch_finish:
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
        log.info("thread finished, pass id: %d, exit" % pass_id)


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
        t = threading.Thread(target=self.preload_thread, args=(dataset_list, ))
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
                dataset.release_memory()
                self.could_load_sem.release()
                continue

            beginpass_begin = time.time()
            self.embedding.begin_pass()
            beginpass_end = time.time()
            log.info("train pass[%d] STAGE [BEGIN PASS] finished, time cost: %f sec" \
                    % (pass_id, beginpass_end - beginpass_begin))

            yield dataset

            dataset.release_memory()
            endpass_begin = time.time()
            self.embedding.end_pass()
            endpass_end = time.time()
            log.info("train pass[%d] STAGE [END PASS] finished, time cost: %f sec" \
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
        t = threading.Thread(target=self.preload_thread, args=(dataset_list, ))
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

            infer_file_num = "%03d" % pass_id
            opt_info = self.infer_model_dict.train_program._fleet_opt
            opt_info["user_define_dump_filename"] = infer_file_num

            beginpass_begin = time.time()
            self.embedding.begin_pass()
            beginpass_end = time.time()
            log.info(
                "infer pass[%d] STAGE [BEGIN PASS] finished, time cost: %f sec",
                pass_id, beginpass_end - beginpass_begin)

            yield dataset

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
