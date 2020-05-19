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

import numpy as np

from paddlerec.core.reader import Reader
from paddlerec.core.utils import envs


class NumpyRandomInt(object):
    def __init__(self, a, b, buf_size=1000):
        self.idx = 0
        self.buffer = np.random.random_integers(a, b, buf_size)
        self.a = a
        self.b = b

    def __call__(self):
        if self.idx == len(self.buffer):
            self.buffer = np.random.random_integers(self.a, self.b,
                                                    len(self.buffer))
            self.idx = 0

        result = self.buffer[self.idx]
        self.idx += 1
        return result


class TrainReader(Reader):
    def init(self):
        dict_path = envs.get_global_env("word_count_dict_path", None, "train.reader")
        self.window_size = envs.get_global_env("hyper_parameters.window_size", None, "train.model")
        self.neg_num = envs.get_global_env("hyper_parameters.neg_num", None, "train.model")
        self.with_shuffle_batch = envs.get_global_env("hyper_parameters.with_shuffle_batch", None, "train.model")
        self.random_generator = NumpyRandomInt(1, self.window_size + 1)

        self.cs = None
        if not self.with_shuffle_batch:
            id_counts = []
            word_all_count = 0
            with io.open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word, count = line.split()[0], int(line.split()[1])
                    id_counts.append(count)
                    word_all_count += count
            id_frequencys = [
                float(count) / word_all_count for count in id_counts
            ]
            np_power = np.power(np.array(id_frequencys), 0.75)
            id_frequencys_pow = np_power / np_power.sum()
            self.cs = np.array(id_frequencys_pow).cumsum()

    def get_context_words(self, words, idx):
        """
        Get the context word list of target word.
        words: the words of the current line
        idx: input word index
        window_size: window size
        """
        target_window = self.random_generator()
        start_point = idx - target_window  # if (idx - target_window) > 0 else 0
        if start_point < 0:
            start_point = 0
        end_point = idx + target_window
        targets = words[start_point:idx] + words[idx + 1:end_point + 1]
        return targets

    def generate_sample(self, line):
        def reader():
            word_ids = [w for w in line.split()]
            for idx, target_id in enumerate(word_ids):
                context_word_ids = self.get_context_words(
                    word_ids, idx)
                for context_id in context_word_ids:
                    output = [('input_word', [int(target_id)]), ('true_label', [int(context_id)])]
                    if not self.with_shuffle_batch:
                        neg_array = self.cs.searchsorted(np.random.sample(self.neg_num))
                        output += [('neg_label', [int(str(i)) for i in neg_array])]
                    yield output

        return reader
