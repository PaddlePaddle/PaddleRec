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

import argparse
import sys
import time
import math
import numpy as np
import six
import paddle
import utils

paddle.enable_static()
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf-8')


def parse_args():
    parser = argparse.ArgumentParser("PaddlePaddle Word2vec infer example")
    parser.add_argument(
        '--dict_path',
        type=str,
        default='./data/data_c/1-billion_dict_word_to_id_',
        help="The path of dic")
    parser.add_argument(
        '--test_dir', type=str, default='test_data', help='test file address')
    parser.add_argument(
        '--print_step', type=int, default='500000', help='print step')
    parser.add_argument(
        '--start_index', type=int, default='0', help='start index')
    parser.add_argument(
        '--last_index', type=int, default='100', help='last index')
    parser.add_argument(
        '--model_dir', type=str, default='model', help='model dir')
    parser.add_argument(
        '--use_cuda', type=int, default='0', help='whether use cuda')
    parser.add_argument(
        '--batch_size', type=int, default='5', help='batch_size')
    parser.add_argument(
        '--emb_size', type=int, default='64', help='batch_size')
    args = parser.parse_args()
    return args


def infer_network(vocab_size, emb_size):
    analogy_a = paddle.static.data(
        name="analogy_a", shape=[None], dtype='int64')
    analogy_b = paddle.static.data(
        name="analogy_b", shape=[None], dtype='int64')
    analogy_c = paddle.static.data(
        name="analogy_c", shape=[None], dtype='int64')
    all_label = paddle.static.data(
        name="all_label", shape=[vocab_size], dtype='int64')
    emb_all_label = paddle.static.nn.embedding(
        input=all_label, size=[vocab_size, emb_size], param_attr="emb")

    emb_a = paddle.static.nn.embedding(
        input=analogy_a, size=[vocab_size, emb_size], param_attr="emb")
    emb_b = paddle.static.nn.embedding(
        input=analogy_b, size=[vocab_size, emb_size], param_attr="emb")
    emb_c = paddle.static.nn.embedding(
        input=analogy_c, size=[vocab_size, emb_size], param_attr="emb")
    target = paddle.add(x=paddle.fluid.layers.nn.elementwise_sub(emb_b, emb_a),
                        y=emb_c)
    emb_all_label_l2 = paddle.fluid.layers.l2_normalize(
        x=emb_all_label, axis=1)
    dist = paddle.fluid.layers.matmul(
        x=target, y=emb_all_label_l2, transpose_y=True)
    values, pred_idx = paddle.topk(x=dist, k=4)
    return values, pred_idx


def infer_epoch(args, vocab_size, test_reader, use_cuda, i2w):
    """ inference function """
    place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    emb_size = args.emb_size
    batch_size = args.batch_size
    with paddle.static.scope_guard(paddle.fluid.Scope()):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            values, pred = infer_network(vocab_size, emb_size)
            for epoch in range(start_index, last_index + 1):
                copy_program = main_program.clone()
                model_path = model_dir + "/" + str(epoch)
                paddle.fluid.io.load_persistables(
                    exe, model_path, main_program=copy_program)
                accum_num = 0
                accum_num_sum = 0.0
                t0 = time.time()
                step_id = 0
                for data in test_reader():
                    step_id += 1
                    b_size = len([dat[0] for dat in data])
                    wa = np.array([dat[0] for dat in data]).astype(
                        "int64").reshape(b_size)
                    wb = np.array([dat[1] for dat in data]).astype(
                        "int64").reshape(b_size)
                    wc = np.array([dat[2] for dat in data]).astype(
                        "int64").reshape(b_size)

                    label = [dat[3] for dat in data]
                    input_word = [dat[4] for dat in data]
                    para = exe.run(copy_program,
                                   feed={
                                       "analogy_a": wa,
                                       "analogy_b": wb,
                                       "analogy_c": wc,
                                       "all_label": np.arange(vocab_size)
                                       .reshape(vocab_size).astype("int64"),
                                   },
                                   fetch_list=[pred.name, values],
                                   return_numpy=False)
                    pre = np.array(para[0])
                    val = np.array(para[1])
                    for ii in range(len(label)):
                        top4 = pre[ii]
                        accum_num_sum += 1
                        for idx in top4:
                            if int(idx) in input_word[ii]:
                                continue
                            if int(idx) == int(label[ii][0]):
                                accum_num += 1
                            break
                    if step_id % 1 == 0:
                        print("step:%d %d " % (step_id, accum_num))

                print("epoch:%d \t acc:%.3f " %
                      (epoch, 1.0 * accum_num / accum_num_sum))


if __name__ == "__main__":
    args = parse_args()
    start_index = args.start_index
    last_index = args.last_index
    test_dir = args.test_dir
    model_dir = args.model_dir
    batch_size = args.batch_size
    dict_path = args.dict_path
    use_cuda = True if args.use_cuda else False
    print("start index: ", start_index, " last_index:", last_index)
    vocab_size, test_reader, id2word = utils.prepare_data(
        test_dir, dict_path, batch_size=batch_size)
    print("vocab_size:", vocab_size)
    infer_epoch(
        args,
        vocab_size,
        test_reader=test_reader,
        use_cuda=use_cuda,
        i2w=id2word)
