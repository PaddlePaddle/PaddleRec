#-*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
"""
    model utils
"""
import os
import sys
import math
import time
import numpy as np
import pickle as pkl

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.static as static
import pgl
from pgl.utils.logger import log
from paddle.common_ops_import import (
    LayerHelper,
    check_type,
    check_variable_and_dtype, )
from paddle.fluid.framework import Variable


def inner_add(value, var):
    """ inner add """
    tmp = var + value
    paddle.assign(tmp, var)


def calc_auc(pos_logits, neg_logits):
    """calc_auc"""
    pos_logits = paddle.reshape(pos_logits[:, 0], [-1, 1])
    neg_logits = paddle.reshape(neg_logits[:, 0], [-1, 1])
    proba = paddle.concat([pos_logits, neg_logits], 0)
    proba = paddle.concat([proba * -1 + 1, proba], axis=1)

    pos_labels = paddle.ones_like(pos_logits)
    neg_labels = paddle.zeros_like(neg_logits)

    pos_labels = paddle.cast(pos_labels, dtype="int64")
    neg_labels = paddle.cast(neg_labels, dtype="int64")

    labels = paddle.concat([pos_labels, neg_labels], 0)
    labels.stop_gradient = True
    _, batch_auc_out, state_tuple = static.auc(input=proba,
                                               label=labels,
                                               num_thresholds=4096)
    return batch_auc_out, state_tuple


def dump_func(file_obj, node_index, node_embed):
    """paddle dump embedding

    输入:

        file_obj: 文件锁对象
        node_index: 节点ID
        node_embed: 节点embedding
        写出路径： output_path/part-%s worker-index
        写出格式: index \t label1 \t label1_score \t label2 \t label2_score

    """
    out = static.default_main_program().current_block().create_var(
        name="dump_out", dtype="float32", shape=[1])

    def _dump_func(vec_id, node_vec):
        """
            Dump Vectors for inference
        """
        if True:
            buffer
            file_obj.acquire()
            vec_lines = []
            for _node_id, _vec_feat in zip(
                    np.array(vec_id), np.array(node_vec)):
                _node_id = str(_node_id.astype("int64").astype('uint64')[0])
                _vec_feat = " ".join(["%.5lf" % w for w in _vec_feat])

                vec_lines.append("%s\t%s\n" % (_node_id, _vec_feat))

            if len(vec_lines) > 0:
                file_obj.vec_path.write(''.join(vec_lines))
                vec_lines = []
            file_obj.release()
        return np.array([1], dtype="float32")

    o = static.py_func(_dump_func, [node_index, node_embed], out=out)
    return o


def paddle_print(*args):
    """print auc"""
    global print_count
    print_count = 0

    global start_time
    start_time = time.time()

    def _print(*inputs):
        """print auc by batch"""
        global print_count
        global start_time
        print_count += 1
        print_per_step = 1000
        if print_count % print_per_step == 0:
            speed = 1.0 * (time.time() - start_time) / print_per_step
            msg = "Speed %s sec/batch \t Batch:%s\t " % (speed, print_count)
            for x in inputs:
                msg += " Loss:%s \t" % (np.array(x)[0])
            log.info(msg)
            start_time = time.time()

    static.py_func(_print, args, out=None)


def loss_visualize(loss):
    """loss_visualize"""
    visualize_loss = static.create_global_var(
        persistable=True, dtype="float32", shape=[1], value=0)
    batch_count = static.create_global_var(
        persistable=True, dtype="float32", shape=[1], value=0)
    inner_add(loss, visualize_loss)
    inner_add(1., batch_count)

    return visualize_loss, batch_count


def build_node_holder(nodeid_slot_name):
    """ build node holder """
    holder_list = []
    nodeid_slot_holder = static.data(
        str(nodeid_slot_name), shape=[-1, 1], dtype="int64", lod_level=1)

    show = static.data("show", shape=[-1], dtype="int64")
    click = static.data("click", shape=[-1], dtype="int64")
    show_clk = paddle.concat(
        [paddle.reshape(show, [-1, 1]), paddle.reshape(click, [-1, 1])],
        axis=-1)
    show_clk = paddle.cast(show_clk, dtype="float32")
    show_clk.stop_gradient = True
    holder_list = [nodeid_slot_holder, show, click]

    return nodeid_slot_holder, show_clk, holder_list


def build_slot_holder(discrete_slot_names):
    """build discrete slot holders """
    holder_list = []
    discrete_slot_holders = []
    discrete_slot_lod_holders = []
    for slot in discrete_slot_names:
        holder = static.data(slot, shape=[None, 1], dtype="int64", lod_level=1)
        discrete_slot_holders.append(holder)
        holder_list.append(holder)

        lod_holder = static.data(
            "slot_%s_lod" % slot, shape=[None], dtype="int64", lod_level=0)
        discrete_slot_lod_holders.append(lod_holder)
        holder_list.append(lod_holder)

    return discrete_slot_holders, discrete_slot_lod_holders, holder_list


def build_token_holder(token_slot_name):
    """build token slot holder """
    token_slot_name = str(token_slot_name)
    token_slot_holder = static.data(
        token_slot_name, shape=[None, 1], dtype="int64", lod_level=1)

    token_slot_lod_holder = static.data(
        "slot_%s_lod" % token_slot_name,
        shape=[None],
        dtype="int64",
        lod_level=0)

    return token_slot_holder, token_slot_lod_holder


def build_graph_holder(samples, use_degree_norm=False):
    """ build graph holder """
    holder_list = []
    graph_holders = {}
    for i, s in enumerate(samples):
        # For different sample size, we hold a graph block.
        graph_holders[i] = []
        num_nodes = static.data(
            name="%s_num_nodes" % i, shape=[-1], dtype="int")
        graph_holders[i].append(num_nodes)
        holder_list.append(num_nodes)

        next_num_nodes = static.data(
            name="%s_next_num_nodes" % i, shape=[-1], dtype="int")
        graph_holders[i].append(next_num_nodes)
        holder_list.append(next_num_nodes)

        edges_src = static.data(
            name="%s_edges_src" % i, shape=[-1, 1], dtype="int64")
        graph_holders[i].append(edges_src)
        holder_list.append(edges_src)

        edges_dst = static.data(
            name="%s_edges_dst" % i, shape=[-1, 1], dtype="int64")
        graph_holders[i].append(edges_dst)
        holder_list.append(edges_dst)

        edges_split = static.data(
            name="%s_edges_split" % i, shape=[-1], dtype="int")
        graph_holders[i].append(edges_split)
        holder_list.append(edges_split)

    ego_index_holder = static.data(name="final_index", shape=[-1], dtype="int")
    holder_list.append(ego_index_holder)

    if use_degree_norm:
        node_degree_holder = static.data(
            name="node_degree", shape=[-1], dtype="int32")
        holder_list.append(node_degree_holder)
    else:
        node_degree_holder = None

    holder_dict = {
        "graph_holders": graph_holders,
        "final_index": ego_index_holder,
        "holder_list": holder_list,
        "node_degree": node_degree_holder
    }

    return holder_dict


def get_sparse_embedding(config,
                         nodeid_slot_holder,
                         discrete_slot_holders,
                         discrete_slot_lod_holders,
                         show_clk,
                         use_cvm,
                         emb_size,
                         name="embedding"):
    """get sparse embedding"""

    id_embedding = static.nn.sparse_embedding(
        input=nodeid_slot_holder,
        size=[1024, emb_size + 3],
        param_attr=paddle.ParamAttr(name=name))

    id_embedding = static.nn.continuous_value_model(id_embedding, show_clk,
                                                    use_cvm)
    id_embedding = id_embedding[:, 1:]  # the first column is for lr, remove it

    tmp_slot_emb_list = []
    for slot_idx, lod in zip(discrete_slot_holders, discrete_slot_lod_holders):
        slot_emb = static.nn.sparse_embedding(
            input=slot_idx,
            size=[1024, emb_size + 3],
            param_attr=paddle.ParamAttr(name=name))

        lod = paddle.cast(lod, dtype="int32")
        lod = paddle.reshape(lod, [1, -1])
        lod.stop_gradient = True
        slot_emb = lod_reset(slot_emb, lod)

        tmp_slot_emb_list.append(slot_emb)

    slot_embedding_list = []
    if (len(discrete_slot_holders)) > 0:
        slot_bows = F.contrib.layers.fused_seqpool_cvm(
            tmp_slot_emb_list,
            config.slot_pool_type,
            show_clk,
            use_cvm=use_cvm)
        for bow in slot_bows:
            slot_embedding = bow[:, 1:]
            slot_embedding = paddle.nn.functional.softsign(slot_embedding)
            slot_embedding_list.append(slot_embedding)

    return id_embedding, slot_embedding_list


def get_degree_norm(degree):
    """ calculate degree normalization """
    degree = paddle.cast(degree, dtype="float32")
    norm = paddle.clip(degree, min=1.0)
    norm = paddle.pow(norm, -0.5)
    norm = paddle.reshape(norm, [-1, 1])
    return norm


def get_graph_degree_norm(graph):
    """ calculate graph degree normalization """
    degree = paddle.zeros(shape=[graph.num_nodes], dtype="int64")
    v, u = graph.edges[:, 0], graph.edges[:, 1]
    degree = paddle.scatter(
        x=degree,
        overwrite=False,
        index=u,
        updates=paddle.ones_like(
            u, dtype="int64"))
    norm = get_degree_norm(degree)
    return norm


def dump_embedding(config, nfeat, node_index):
    """dump_embedding"""
    node_embed = paddle.squeeze(
        nfeat, axis=[1], name=config.dump_node_emb_name)
    node_index = paddle.reshape(node_index, shape=[-1, 2])
    src_node_index = node_index[:, 0:1]
    src_node_index = paddle.reshape(
        src_node_index, shape=src_node_index.shape,
        name=config.dump_node_name)  # for rename


def hcl(config, feature, graph_holders):
    """Hierarchical Contrastive Learning"""
    hcl_logits = []
    for idx, sample in enumerate(config.samples):
        graph_holder = graph_holders[idx]
        edges_src = graph_holder[2]
        edges_dst = graph_holder[3]
        neighbor_src_feat = paddle.gather(feature, edges_src)
        neighbor_src_feat = neighbor_src_feat.reshape([-1, 1, config.emb_size])
        neighbor_dst_feat = paddle.gather(feature, edges_dst)
        neighbor_dst_feat = neighbor_dst_feat.reshape([-1, 1, config.emb_size])
        neighbor_dsts_feat_all = [neighbor_dst_feat]

        for neg in range(config.neg_num):
            neighbor_dsts_feat_all.append(
                F.contrib.layers.shuffle_batch(neighbor_dsts_feat_all[0]))
        neighbor_dsts_feat = paddle.concat(neighbor_dsts_feat_all, axis=1)

        # [batch_size, 1, neg_num+1]
        logits = paddle.matmul(
            neighbor_src_feat, neighbor_dsts_feat, transpose_y=True)
        logits = paddle.squeeze(logits, axis=[1])
        hcl_logits.append(logits)

    return hcl_logits


def reset_program_state_dict(args, model, state_dict, pretrained_state_dict):
    """
    Initialize the parameter from the bert config, and set the parameter by 
    reseting the state dict."
    """
    scale = args.init_range
    reset_state_dict = {}
    reset_parameter_names = []
    for n, p in state_dict.items():
        if n in pretrained_state_dict:
            log.info("p_name: %s , pretrained name: %s" % (p.name, n))
            reset_state_dict[p.name] = np.array(pretrained_state_dict[n])
            reset_parameter_names.append(n)
        #  elif p.name in pretrained_state_dict and "bert" in n:
        #      reset_state_dict[p.name] = np.array(pretrained_state_dict[p.name])
        #      reset_parameter_names.append(n)
        else:
            log.info("[RANDOM] p_name: %s , pretrained name: %s" % (p.name, n))
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            reset_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
    log.info("the following parameter had reset, please check. {}".format(
        reset_parameter_names))
    return reset_state_dict


def lod_reset(x, y=None, target_lod=None):
    """lod_reset"""
    check_variable_and_dtype(x, 'x', ['float32', 'float64', 'int32', 'int64'],
                             'lod_reset')
    h = LayerHelper("lod_reset", **locals())
    out = h.create_variable_for_type_inference(dtype=x.dtype)
    if y is not None:
        check_type(y, 'y', (Variable), 'lod_reset')
        # TODO: check y.lod_level = 0 dtype
        h.append_op(
            type="lod_reset", inputs={'X': x,
                                      'Y': y}, outputs={'Out': out})
    elif target_lod is not None:
        h.append_op(
            type="lod_reset",
            inputs={'X': x},
            attrs={'target_lod': target_lod},
            outputs={'Out': out}, )
    else:
        raise ValueError("y and target_lod should not be both none.")
    return out
