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
"""Distributed GPU Graph config
"""
import sys
import time
import paddle
from pgl.utils.logger import log

import helper
import util
from place import get_cuda_places


class DistGraph(object):
    """ Initialize the Distributed Graph Server

    Args:
        root_dir: the graph data dir 
    
        node_types: the node type configs
 
        edge_types: the edge type configs.

        symmetry: whether the edges are symmetry

        slots: the node feature slot 

        token_slot: for erniesage token slot input

        slot_num_for_pull_feature: total slot feature number we should pull

        num_parts: the sharded parts of graph data

        metapath_split_opt: whether use metapath split optimization
    """

    def __init__(self,
                 root_dir,
                 node_types,
                 edge_types,
                 symmetry,
                 slots,
                 token_slot,
                 float_slots,
                 float_slots_len,
                 slot_num_for_pull_feature,
                 float_slot_num,
                 num_parts,
                 metapath_split_opt=False,
                 train_start_nodes=None,
                 infer_nodes=None,
                 use_weight=False):

        self.root_dir = root_dir
        self.node_types = node_types
        self.edge_types = edge_types
        self.symmetry = symmetry
        self.slots = slots
        self.token_slot = token_slot
        self.float_slots = float_slots
        self.float_slots_len = float_slots_len
        self.slot_num_for_pull_feature = slot_num_for_pull_feature
        self.float_slot_num = float_slot_num
        self.num_parts = num_parts
        self.metapath_split_opt = metapath_split_opt
        self.train_start_nodes = train_start_nodes
        self.infer_nodes = infer_nodes
        self.use_weight = use_weight
        self.reverse = 1 if self.symmetry else 0

        self.etype2files = helper.parse_files(self.edge_types)
        self.etype_list = util.get_all_edge_type(self.etype2files,
                                                 self.symmetry)

        self.ntype2files = helper.parse_files(self.node_types)
        self.ntype_list = list(self.ntype2files.keys())
        log.info("total etype: %s" % repr(self.etype_list))
        log.info("total ntype: %s" % repr(self.ntype_list))

        self.graph = paddle.framework.core.GraphGpuWrapper()

        self._setup_graph()

    def _setup_graph(self):
        self.graph.set_device(get_cuda_places())
        self.graph.set_up_types(self.etype_list, self.ntype_list)

        for ntype in self.ntype_list:
            for slot_id in self.slots:
                log.info("add_table_feat_conf of slot id %s" % slot_id)
                self.graph.add_table_feat_conf(ntype,
                                               str(slot_id), "feasign", 1)

            if self.token_slot:
                log.info("add_table_feat_conf of token slot id %s" %
                         self.token_slot)
                self.graph.add_table_feat_conf(ntype,
                                               str(self.token_slot), "feasign",
                                               1)
            if self.float_slots:
                for float_slot_id in self.float_slots:
                    float_slot_len = self.float_slots_len[str(float_slot_id)]
                    log.info("add_table_feat_conf of float slot id %s" % float_slot_id)
                    self.graph.add_table_feat_conf(ntype,
                                               str(float_slot_id), "float32", int(float_slot_len))

        self.graph.set_slot_feature_separator(":")
        self.graph.set_feature_separator(",")
        self.graph.init_service()

    def load_edge_serial(self):
        """load edge serially, just used in metapath_split_opt
        """
        for etype in self.etype_list:
            if etype not in self.etype2files:
                continue
            load_begin_time = time.time()
            self.graph.load_edge_file(etype + ":" + etype, self.root_dir,
                                      self.num_parts, self.reverse, [],
                                      self.use_weight)
            load_end_time = time.time()
            log.info("load edge[%s] to cpu, time cost: %f sec" %
                     (etype, load_end_time - load_begin_time))

            release_begin_time = time.time()
            self.graph.release_graph_edge()
            release_end_time = time.time()
            log.info("STAGE [CPU RELEASE EDGE] finished, time cost: %f sec" %
                     (release_end_time - release_begin_time))

    def load_edge(self):
        """Pull whole graph edges from disk into cpu memory, then load into gpu.
           After that, release memory on cpu.
        """
        if self.metapath_split_opt:
            return self.load_edge_serial()

        load_begin_time = time.time()
        self.graph.load_edge_file(self.edge_types, self.root_dir,
                                  self.num_parts, self.reverse, [],
                                  self.use_weight)
        load_end_time = time.time()
        log.info("STAGE [CPU LOAD EDGE] finished, time cost: %f sec" %
                 (load_end_time - load_begin_time))

        log.info("begin calc edge neighbor limit")
        self.graph.calc_edge_type_limit()
        log.info("end calc edge neighbor limit")

        load_begin_time = time.time()
        for i in range(len(self.etype_list)):
            log.info("begin to upload edge_type: [%s] to GPU" %
                     self.etype_list[i])
            self.graph.upload_batch(0, len(get_cuda_places()), self.etype_list[i])
        load_end_time = time.time()
        log.info("STAGE [GPU LOAD EDGE] finished, time cost: %f sec" %
                 (load_end_time - load_begin_time))

        release_begin_time = time.time()
        self.graph.release_graph_edge()
        release_end_time = time.time()
        log.info("STAGE [CPU RELEASE EDGE] finished, time cost: %f sec" %
                 (release_end_time - release_begin_time))

    def load_node(self):
        """Pull whole graph nodes from disk into cpu memory, then load into gpu.
           After that, release memory on cpu.
        """
        load_begin_time = time.time()
        ret = self.graph.load_node_file(self.node_types, self.root_dir,
                                        self.num_parts)

        if self.train_start_nodes:
            log.info("[TRAIN NODE] loading train_start_nodes from [%s]" % self.train_start_nodes)
            new_ntype2files = helper.generate_files_string(
                    self.ntype_list, self.train_start_nodes)
            log.info("[TRAIN NODE] train_node2files: %s" % new_ntype2files)
            ret = self.graph.load_node_file(new_ntype2files, self.root_dir,
                                            self.num_parts)
            if ret != 0:
                log.info("Fail to load train_start_nodes!")
                return -1

        if self.infer_nodes:
            log.info("[INFER NODE] loading infer_nodes from [%s]" % self.infer_nodes)

            new_ntype2files = helper.generate_files_string(
                    self.ntype_list, self.infer_nodes)
            log.info("[INFER NODE] infer_node2files: %s" % new_ntype2files)

            ret = self.graph.load_node_file(new_ntype2files, self.root_dir,
                                            self.num_parts)
            if ret != 0:
                log.info("Fail to load infer_nodes!")
                return -1

        if ret != 0:
            log.info("Fail to load node, ntype2files[%s] path[%s] num_part[%d]" \
                     % (self.node_types, self.root_dir, self.num_parts))
            self.graph.release_graph_node()
            return -1

        load_end_time = time.time()
        log.info("STAGE [CPU LOAD NODE] finished, time cost: %f sec" %
                 (load_end_time - load_begin_time))

        if not self.metapath_split_opt:
            load_begin_time = time.time()
            if self.slot_num_for_pull_feature > 0 or self.float_slot_num > 0:
                self.graph.upload_batch(1,
                        len(get_cuda_places()),
                        self.slot_num_for_pull_feature, self.float_slot_num)
            load_end_time = time.time()
            log.info("STAGE [GPU LOAD NODE] finished, time cost: %f sec" %
                    (load_end_time - load_begin_time))

        release_begin_time = time.time()
        self.graph.release_graph_node()
        release_end_time = time.time()
        log.info("STAGE [CPU RELEASE NODE] finished, time cost: %f sec" %
                 (release_end_time - release_begin_time))

        return 0

    def load_metapath_edges_nodes(self, metapath_dict, metapath, i):
        """Pull specific metapath's edges.
        """
        log.info("Begin load_metapath_edges_nodes, metapath[%d]: %s" %
                 (i, metapath))
        first_node = metapath.split('2')[0]
        all_metapath_class_len = len(metapath_dict[first_node])
        cur_metapath_index = metapath_dict[first_node].index(metapath)

        sub_edges_list = metapath.split('-')
        edge_len = len(sub_edges_list)
        sub_etype2files, is_reverse_map = util.get_sub_path(
            self.etype2files, sub_edges_list, True)
        log.info("metapath[%s] etype2files[%s] is_reverse_map[%s]", metapath,
                 sub_etype2files, is_reverse_map)

        metapath_cpuload_begin = time.time()
        self.graph.load_edge_file(sub_etype2files, self.root_dir,
                                  self.num_parts, False, is_reverse_map,
                                  self.use_weight)
        metapath_cpuload_end = time.time()
        log.info("metapath[%s] load edges[%s] to cpu, time: %s" %
                 (metapath, sub_etype2files,
                  metapath_cpuload_end - metapath_cpuload_begin))

        metapath_gpuload_begin = time.time()
        for j in range(0, len(sub_edges_list)):
            self.graph.upload_batch(0,
                                    len(get_cuda_places()), sub_edges_list[j])
        metapath_gpuload_end = time.time()
        log.info("metapath[%s] load edges[%s] to gpu, time: %s" %
                 (metapath, sub_etype2files,
                  metapath_gpuload_end - metapath_gpuload_begin))

        self.graph.release_graph_edge()

        sub_nodes_list = []
        for edge in sub_edges_list:
            sub_nodes_list.extend(edge.split('2'))
        sub_ntype2files, _ = util.get_sub_path(self.ntype2files,
                                               sub_nodes_list, False)
        log.info("metapath[%s] ntype2files[%s]", metapath, sub_ntype2files)
        load_begin_time = time.time()
        ret = self.graph.load_node_file(sub_ntype2files, self.root_dir,
                                        self.num_parts)
        if ret != 0:
            log.info("Fail to load node, ntype2files[%s] path[%s] num_part[%d]" \
                     % (sub_ntype2files, self.root_dir, self.num_parts))
            self.graph.release_graph_node()
            return -1
        load_end_time = time.time()
        log.info("metapath[%s] load nodes[%s] to cpu, time: %s" %
                 (metapath, sub_ntype2files, load_end_time - load_begin_time))

        load_begin_time = time.time()
        if self.slot_num_for_pull_feature > 0:
            self.graph.upload_batch(1,
                                    len(get_cuda_places()),
                                    self.slot_num_for_pull_feature)
        load_end_time = time.time()
        log.info("metapath[%s] load nodes[%s] to gpu, time: %s" %
                 (metapath, sub_ntype2files, load_end_time - load_begin_time))

        self.graph.release_graph_node()

        self.graph.init_metapath(metapath, cur_metapath_index,
                                 all_metapath_class_len)

    def get_sorted_metapath_and_dict(self, metapaths):
        """doc"""
        first_node_type = util.get_first_node_type(metapaths)
        node_type_size = self.graph.get_node_type_size(first_node_type)
        edge_type_size = self.graph.get_edge_type_size()
        sorted_metapaths = util.change_metapath_index(
            metapaths, node_type_size, edge_type_size)
        log.info("after change metapaths: %s" % sorted_metapaths)

        metapath_dict = {}
        for i in range(len(sorted_metapaths)):
            first_node = sorted_metapaths[i].split('2')[0]
            if first_node in metapath_dict:
                metapath_dict[first_node].append(sorted_metapaths[i])
            else:
                metapath_dict[first_node] = []
                metapath_dict[first_node].append(sorted_metapaths[i])
        return sorted_metapaths, metapath_dict

    def clear_metapath_state(self):
        """doc"""
        self.graph.clear_metapath_state()

    def load_train_node_from_file(self, train_start_nodes_path_vec, is_need_shuffle):
        """ load train node from file"""
        if not train_start_nodes_path_vec:
            self.graph.set_node_iter_from_graph(True, is_need_shuffle)
        else:
            nodetype2files_vec = helper.generate_files_string(
                self.ntype_list, train_start_nodes_path_vec)
            self.graph.set_node_iter_from_file(
                nodetype2files_vec, self.root_dir, self.num_parts, True, is_need_shuffle)
    def load_infer_node_from_file(self, infer_nodes_path_vec):
        """ load infer node from file"""
        if not infer_nodes_path_vec:
            self.graph.set_node_iter_from_graph(False, False)
        else:
            # for online learning, maybe we need to change input for dir.
            nodetype2files_vec = helper.generate_files_string(
                self.ntype_list, infer_nodes_path_vec)
            self.graph.set_node_iter_from_file(
                nodetype2files_vec, self.root_dir, self.num_parts, False, False)

    def load_graph_into_cpu(self):
        """Pull whole graph from disk into memory
        """
        cpuload_begin = time.time()
        self.graph.load_node_and_edge(self.edge_types, self.node_types,
                                      self.root_dir, self.num_parts,
                                      self.reverse)
        cpuload_end = time.time()
        log.info("STAGE [CPU LOAD] finished, time cost: %f sec",
                 cpuload_end - cpuload_begin)

    def load_graph_into_gpu(self):
        """Pull whole graph from memory into gpu 
        """
        gpuload_begin = time.time()
        log.info("STAGE [GPU Load] begin load edges from cpu to gpu")
        for i in range(len(self.etype_list)):
            self.graph.upload_batch(0,
                                    len(get_cuda_places()), self.etype_list[i])
            log.info("STAGE [GPU Load] end load edge into GPU, type[" +
                     self.etype_list[i] + "]")

        slot_num = len(self.slots)
        log.info("STAGE [GPU Load] begin load node from cpu to gpu")
        if slot_num > 0:
            self.graph.upload_batch(1, len(get_cuda_places()), slot_num)
        log.info("STAGE [GPU Load] end load node from cpu to gpu")
        gpuload_end = time.time()
        log.info("STAGE [GPU LOAD] finished, time cost: %f sec",
                 gpuload_end - gpuload_begin)

    def finalize(self):
        """release the graph"""
        self.graph.finalize()

    def __del__(self):
        self.finalize()
