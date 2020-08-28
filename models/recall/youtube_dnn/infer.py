# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import copy
import numpy as np
import argparse
import paddle.fluid as fluid
import pandas as pd
from paddle.fluid.incubate.fleet.utils import utils


def parse_args():
    parser = argparse.ArgumentParser("PaddlePaddle Youtube DNN infer example")
    parser.add_argument(
        '--use_gpu', type=int, default='0', help='whether use gpu')
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument(
        "--test_epoch", type=int, default=19, help="test_epoch")
    parser.add_argument(
        '--inference_model_dir',
        type=str,
        default='./inference_youtubednn',
        help='inference_model_dir')
    parser.add_argument(
        '--increment_model_dir',
        type=str,
        default='./increment_youtubednn',
        help='persistable_model_dir')
    parser.add_argument(
        '--watch_vec_size', type=int, default=64, help='watch_vec_size')
    parser.add_argument(
        '--search_vec_size', type=int, default=64, help='search_vec_size')
    parser.add_argument(
        '--other_feat_size', type=int, default=64, help='other_feat_size')
    parser.add_argument('--topk', type=int, default=5, help='topk')
    args = parser.parse_args()
    return args


def infer(args):
    video_save_path = os.path.join(args.increment_model_dir,
                                   str(args.test_epoch), "l4_weight")
    video_vec, = utils.load_var("l4_weight", [32, 100], 'float32',
                                video_save_path)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    cur_model_path = os.path.join(args.inference_model_dir,
                                  str(args.test_epoch))

    user_vec = None
    with fluid.scope_guard(fluid.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(
            cur_model_path, exe)
        # Build a random data set.
        sample_size = 100
        watch_vecs = []
        search_vecs = []
        other_feats = []

        for i in range(sample_size):
            watch_vec = np.random.rand(1, args.watch_vec_size)
            search_vec = np.random.rand(1, args.search_vec_size)
            other_feat = np.random.rand(1, args.other_feat_size)
            watch_vecs.append(watch_vec)
            search_vecs.append(search_vec)
            other_feats.append(other_feat)

        for i in range(sample_size):
            l3 = exe.run(infer_program,
                         feed={
                             "watch_vec": watch_vecs[i].astype('float32'),
                             "search_vec": search_vecs[i].astype('float32'),
                             "other_feat": other_feats[i].astype('float32'),
                         },
                         return_numpy=True,
                         fetch_list=fetch_vars)
            if user_vec is not None:
                user_vec = np.concatenate([user_vec, l3[0]], axis=0)
            else:
                user_vec = l3[0]

    # get topk result
    user_video_sim_list = []
    for i in range(user_vec.shape[0]):
        for j in range(video_vec.shape[1]):
            user_video_sim = cos_sim(user_vec[i], video_vec[:, j])
            user_video_sim_list.append(user_video_sim)

        tmp_list = copy.deepcopy(user_video_sim_list)
        tmp_list.sort()
        max_sim_index = [
            user_video_sim_list.index(one)
            for one in tmp_list[::-1][:args.topk]
        ]

        print("user:{0}, top K videos:{1}".format(i, max_sim_index))
        user_video_sim_list = []


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / (denom + 1e-4)
    sim = 0.5 + 0.5 * cos
    return sim


if __name__ == "__main__":
    args = parse_args()
    infer(args)
