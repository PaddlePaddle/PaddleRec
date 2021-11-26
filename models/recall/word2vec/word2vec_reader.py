#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import numpy as np
import io
import six

from paddle.io import IterableDataset


class NumpyRandomInt(object):
    def __init__(self, a, b, buf_size=1000):
        self.idx = 0
        np.random.seed(12345)
        self.buffer = np.random.randint(a, b, buf_size)
        self.a = a
        self.b = b

    def __call__(self):
        if self.idx == len(self.buffer):
            np.random.seed(12345)
            self.buffer = np.random.randint(self.a, self.b, len(self.buffer))
            self.idx = 0

        result = self.buffer[self.idx]
        self.idx += 1
        return result


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.config = config
        self.init()

    def init(self):
        dict_path = self.config.get("runner.word_count_dict_path")
        self.window_size = self.config.get("hyper_parameters.window_size")
        self.neg_num = self.config.get("hyper_parameters.neg_num")
        self.with_shuffle_batch = self.config.get(
            "hyper_parameters.with_shuffle_batch")
        self.random_generator = NumpyRandomInt(1, self.window_size + 1)
        self.batch_size = self.config.get("runner.batch_size")

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
        # if (idx - target_window) > 0 else 0
        start_point = idx - target_window
        if start_point < 0:
            start_point = 0
        end_point = idx + target_window
        targets = words[start_point:idx] + words[idx + 1:end_point + 1]
        return targets

    def __iter__(self):
        full_lines = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for line in rf:
                    word_ids = [w for w in line.split()]
                    for idx, target_id in enumerate(word_ids):
                        context_word_ids = self.get_context_words(word_ids,
                                                                  idx)
                        for context_id in context_word_ids:
                            output = []
                            output.append(
                                np.array([int(target_id)]).astype('int64'))
                            output.append(
                                np.array([int(context_id)]).astype('int64'))
                            np.random.seed(12345)
                            neg_array = self.cs.searchsorted(
                                np.random.sample(self.neg_num))
                            output.append(
                                np.array([int(str(i))
                                          for i in neg_array]).astype('int64'))
                            yield output


class Word2VecInferDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(Word2VecInferDataset, self).__init__()
        self.file_list = file_list
        self.config = config
        self.init()

    def init(self):
        dict_path = self.config.get("runner.word_id_dict_path")
        self.word_to_id = dict()
        self.id_to_word = dict()
        with io.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
                self.id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]
        self.dict_size = len(self.word_to_id)

    def native_to_unicode(self, s):
        if self._is_unicode(s):
            return s
        try:
            return self._to_unicode(s)
        except UnicodeDecodeError:
            res = self._to_unicode(s, ignore_errors=True)
            return res

    def _is_unicode(self, s):
        if six.PY2:
            if isinstance(s, unicode):
                return True
        else:
            if isinstance(s, str):
                return True
        return False

    def _to_unicode(self, s, ignore_errors=False):
        if self._is_unicode(s):
            return s
        error_mode = "ignore" if ignore_errors else "strict"
        return s.decode("utf-8", errors=error_mode)

    def strip_lines(self, line, vocab):
        return self._replace_oov(vocab, self.native_to_unicode(line))

    def _replace_oov(self, original_vocab, line):
        """Replace out-of-vocab words with "<UNK>".
      This maintains compatibility with published results.
      Args:
        original_vocab: a set of strings (The standard vocabulary for the dataset)
        line: a unicode string - a space-delimited sequence of words.
      Returns:
        a unicode string - a space-delimited sequence of words.
      """
        return u" ".join([
            word if word in original_vocab else u"<UNK>"
            for word in line.split()
        ])

    def __iter__(self):
        full_lines = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for line in rf:
                    if ':' in line:
                        return
                    features = self.strip_lines(line.lower(), self.word_to_id)
                    features = features.split()
                    output_list = []
                    for i in range(4):
                        output_list.append(
                            np.array([self.word_to_id[features[i]]]).astype(
                                'int64'))
                    inputs_words = [
                        self.word_to_id[features[i]] for i in range(3)
                    ]
                    output_list.append(np.array(inputs_words).astype('int64'))
                    yield output_list
