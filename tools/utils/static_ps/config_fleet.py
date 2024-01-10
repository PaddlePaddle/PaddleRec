# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Generate Proto for pslib
"""
import os
import copy

import numpy as np
import paddle
from pgl.utils.logger import log


def get_strategy(args, model_dict):
    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.a_sync = True  # 默认使用async模式

    configs = {"use_ps_gpu": 1}
    strategy.a_sync_configs = configs

    # sparse参数相关配置
    strategy.fleet_desc_configs = generate_config(args)

    if args.fs_name or args.fs_ugi:
        user, passwd = args.fs_ugi.split(',', 1)
        strategy.fs_client_param = {
            "uri": args.fs_name,
            "user": user,
            "passwd": passwd,
            "hadoop_bin": "%s/bin/hadoop" % (os.getenv("HADOOP_HOME"))
        }
        log.info("set DistributedStrategy fs_client_param")
    else:
        strategy.fs_client_param = {
            "uri": "",
            "user": "",
            "passwd": "",
            "hadoop_bin": ""
        }
        log.info("not set DistributedStrategy fs_client_param")

    return strategy


def gen_sparse_config(args, sparse_lr, init_range, op_type, emb_size, feature_lr, \
                     nodeid_slot, load_filter_slots, save_filter_slots, sparse_table_class):
    """
    gen sparse config
    """
    sparse_config = dict()
    sparse_config['sparse_table_class'] = sparse_table_class
    sparse_config['sparse_compress_in_save'] = True
    sparse_config['sparse_shard_num'] = 67
    # sparse_config['sparse_accessor_class'] = "DownpourCtrAccessor"
    sparse_config[
        'sparse_accessor_class'] = "DownpourCtrDymfAccessor"  # for variable embedding
    sparse_config['sparse_learning_rate'] = sparse_lr
    sparse_config['sparse_initial_g2sum'] = 3
    sparse_config['sparse_initial_range'] = init_range
    sparse_config['sparse_weight_bounds'] = [-10.0, 10.0]
    sparse_config['sparse_embedx_dim'] = emb_size
    sparse_config['sparse_embedx_threshold'] = 0
    sparse_config['sparse_nonclk_coeff'] = 0.1
    sparse_config['sparse_click_coeff'] = 1.0
    sparse_config['sparse_base_threshold'] = 0
    sparse_config['sparse_delta_threshold'] = 0.25
    sparse_config['sparse_delta_keep_days'] = 16.0
    sparse_config['sparse_show_click_decay_rate'] = 0.98
    sparse_config['sparse_delete_threshold'] = 0.8
    sparse_config['sparse_delete_after_unseen_days'] = 30

    sparse_config['embed_sparse_optimizer'] = op_type
    sparse_config['embed_sparse_learning_rate'] = sparse_lr
    sparse_config['embed_sparse_initial_range'] = 0
    sparse_config[
        'embed_sparse_beta1_decay_rate'] = 0.9  #args.beta1_decay_rate
    sparse_config[
        'embed_sparse_beta2_decay_rate'] = 0.999  #args.beta2_decay_rate
    sparse_config['embed_sparse_weight_bounds'] = [-10.0, 10.0]

    sparse_config['embedx_sparse_optimizer'] = op_type
    sparse_config['embedx_sparse_learning_rate'] = sparse_lr
    sparse_config['embedx_sparse_initial_range'] = init_range
    sparse_config[
        'embedx_sparse_beta1_decay_rate'] = 0.9  #args.beta1_decay_rate
    sparse_config[
        'embedx_sparse_beta2_decay_rate'] = 0.999  #args.beta2_decay_rate
    sparse_config['embedx_sparse_weight_bounds'] = [-10.0, 10.0]
    sparse_config['nodeid_slot'] = nodeid_slot
    sparse_config['feature_learning_rate'] = feature_lr
    sparse_config['sparse_load_filter_slots'] = load_filter_slots
    sparse_config['sparse_save_filter_slots'] = save_filter_slots
    return sparse_config


def generate_config(args):
    """ Generate Proto For PSlib  """
    config = dict()
    config['use_cvm'] = True
    config['trainer'] = "PSGPUTrainer"
    config['worker_class'] = "PSGPUWorker"
    config['use_ps_gpu'] = True
    # embedding name as key name
    # Id Embedding
    gen_config = gen_sparse_config

    slot_feature_lr = args.sparse_lr
    if "slot_feature_lr" in args:
        slot_feature_lr = args.slot_feature_lr
    if "train_storage_mode" in args and args.train_storage_mode == "SSD_EMBEDDING":
        sparse_table_class = "DownpourSparseSSDTable"
    else:
        sparse_table_class = "DownpourSparseTable"
    config['embedding'] = gen_config(args, args.sparse_lr, args.init_range, args.sparse_type, \
                                     args.emb_size, slot_feature_lr, args.nodeid_slot, args.load_filter_slots, \
                                     args.save_filter_slots, sparse_table_class)

    dense_config = dict()
    dense_config['dense_table_class'] = "DownpourDenseTable"
    dense_config['dense_compress_in_save'] = True
    dense_config['dense_accessor_class'] = "DownpourDenseValueAccessor"
    dense_config['dense_learning_rate'] = args.dense_lr
    dense_config['dense_optimizer'] = "adam"
    dense_config['dense_avg_decay'] = 0.999993
    dense_config['dense_ada_decay'] = 0.9999
    dense_config['dense_ada_epsilon'] = 1e-8
    dense_config['dense_mom_decay'] = 0.99
    dense_config['dense_naive_lr'] = 0.0002
    # 'dense_table' as key name
    config['dense_table'] = dense_config

    datanorm_config = dict()
    datanorm_config['datanorm_table_class'] = "DownpourDenseTable"
    datanorm_config['datanorm_compress_in_save'] = True
    datanorm_config['datanorm_accessor_class'] = "DownpourDenseValueAccessor"
    datanorm_config['datanorm_operation'] = "summary"
    datanorm_config['datanorm_decay_rate'] = 0.999999
    config['datanorm_table'] = datanorm_config

    return config
