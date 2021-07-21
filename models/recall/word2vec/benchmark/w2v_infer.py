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

import io
import os
import argparse
import sys
import time
import math
import numpy as np
import six
import ast
import paddle.fluid as fluid
import paddle
import collections
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf-8')
paddle.enable_static()


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
        '--emb_size', type=int, default='300', help='batch_size')
    parser.add_argument(
        '-bf16',
        '--pure_bf16',
        type=ast.literal_eval,
        default=False,
        help="whether use bf16")
    args = parser.parse_args()
    return args


def infer_network(vocab_size, emb_size, pure_bf16=False):
    analogy_a = fluid.data(name="analogy_a", shape=[None], dtype='int64')
    analogy_b = fluid.data(name="analogy_b", shape=[None], dtype='int64')
    analogy_c = fluid.data(name="analogy_c", shape=[None], dtype='int64')
    all_label = fluid.data(name="all_label", shape=[vocab_size], dtype='int64')
    dtype = 'uint16' if pure_bf16 else 'float32'
    emb_all_label = fluid.embedding(
        input=all_label,
        size=[vocab_size, emb_size],
        param_attr="emb",
        dtype=dtype)

    emb_a = fluid.embedding(
        input=analogy_a,
        size=[vocab_size, emb_size],
        param_attr="emb",
        dtype=dtype)
    emb_b = fluid.embedding(
        input=analogy_b,
        size=[vocab_size, emb_size],
        param_attr="emb",
        dtype=dtype)
    emb_c = fluid.embedding(
        input=analogy_c,
        size=[vocab_size, emb_size],
        param_attr="emb",
        dtype=dtype)

    if pure_bf16:
        emb_all_label = fluid.layers.cast(emb_all_label, "float32")
        emb_a = fluid.layers.cast(emb_a, "float32")
        emb_b = fluid.layers.cast(emb_b, "float32")
        emb_c = fluid.layers.cast(emb_c, "float32")

    target = fluid.layers.elementwise_add(
        fluid.layers.elementwise_sub(emb_b, emb_a), emb_c)

    emb_all_label_l2 = fluid.layers.l2_normalize(x=emb_all_label, axis=1)
    dist = fluid.layers.matmul(x=target, y=emb_all_label_l2, transpose_y=True)
    values, pred_idx = fluid.layers.topk(input=dist, k=4)
    return values, pred_idx


def _load_emb(var):
    res = (var.name == "emb")
    return res


def infer_epoch(args, vocab_size, test_reader, use_cuda, i2w):
    """ inference function """
    epoch_model_path_list = []

    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        # hard code for epoch model folder
        if os.path.isdir(file_path) and is_number(file):
            epoch_model_path_list.append(file_path)

    if len(epoch_model_path_list) == 0:
        return
    epoch_model_path_list.sort()
    print("Save model len {}".format(len(epoch_model_path_list)))

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    emb_size = args.emb_size
    batch_size = args.batch_size
    pure_bf16 = args.pure_bf16

    result_dict = {}
    result_dict["result"] = {}

    with fluid.scope_guard(fluid.Scope()):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            values, pred = infer_network(vocab_size, emb_size, pure_bf16)
            for epoch, model_path in enumerate(epoch_model_path_list):
                print("Begin infer model: {}".format(model_path))
                copy_program = main_program.clone()
                fluid.io.load_vars(
                    executor=exe, dirname=model_path, predicate=_load_emb)
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
                print("model: {} \t acc: {} ".format(
                    model_path, 1.0 * accum_num / accum_num_sum))
                epoch_acc = 1.0 * accum_num / accum_num_sum
                epoch_name = model_path.split("/")[-1]
                result_dict["result"][epoch_name] = epoch_acc

    print("infer_result_dict: {}".format(result_dict))
    with open("./infer_result_dict.txt", 'w+') as f:
        f.write(str(result_dict))


def BuildWord_IdMap(dict_path):
    word_to_id = dict()
    id_to_word = dict()
    with io.open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
            id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]
    return word_to_id, id_to_word


def prepare_data(file_dir, dict_path, batch_size):
    w2i, i2w = BuildWord_IdMap(dict_path)
    vocab_size = len(i2w)
    reader = fluid.io.batch(test(file_dir, w2i), batch_size)
    return vocab_size, reader, i2w


def check_version(with_shuffle_batch=False):
    """
     Log error and exit when the installed version of paddlepaddle is
     not satisfied.
     """
    err = "PaddlePaddle version 1.6 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        if with_shuffle_batch:
            fluid.require_version('1.7.0')
        else:
            fluid.require_version('1.6.0')
    except Exception as e:
        logger.error(err)
        sys.exit(1)


def native_to_unicode(s):
    if _is_unicode(s):
        return s
    try:
        return _to_unicode(s)
    except UnicodeDecodeError:
        res = _to_unicode(s, ignore_errors=True)
        return res


def _is_unicode(s):
    if six.PY2:
        if isinstance(s, unicode):
            return True
    else:
        if isinstance(s, str):
            return True
    return False


def _to_unicode(s, ignore_errors=False):
    if _is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)


def strip_lines(line, vocab):
    return _replace_oov(vocab, native_to_unicode(line))


def _replace_oov(original_vocab, line):
    """Replace out-of-vocab words with "<UNK>".
  This maintains compatibility with published results.
  Args:
    original_vocab: a set of strings (The standard vocabulary for the dataset)
    line: a unicode string - a space-delimited sequence of words.
  Returns:
    a unicode string - a space-delimited sequence of words.
  """
    return u" ".join([
        word if word in original_vocab else u"<UNK>" for word in line.split()
    ])


def reader_creator(file_dir, word_to_id):
    def reader():
        files = os.listdir(file_dir)
        for fi in files:
            with io.open(
                    os.path.join(file_dir, fi), "r", encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        pass
                    else:
                        line = strip_lines(line.lower(), word_to_id)
                        line = line.split()
                        yield [word_to_id[line[0]]], [word_to_id[line[1]]], [
                            word_to_id[line[2]]
                        ], [word_to_id[line[3]]], [
                            word_to_id[line[0]], word_to_id[line[1]],
                            word_to_id[line[2]]
                        ]

    return reader


def test(test_dir, w2i):
    return reader_creator(test_dir, w2i)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


if __name__ == "__main__":
    args = parse_args()
    test_dir = args.test_dir
    model_dir = args.model_dir
    batch_size = args.batch_size
    dict_path = args.dict_path
    use_cuda = True if args.use_cuda else False
    vocab_size, test_reader, id2word = prepare_data(
        test_dir, dict_path, batch_size=batch_size)
    print("vocab_size:", vocab_size)
    infer_epoch(
        args,
        vocab_size,
        test_reader=test_reader,
        use_cuda=use_cuda,
        i2w=id2word)
