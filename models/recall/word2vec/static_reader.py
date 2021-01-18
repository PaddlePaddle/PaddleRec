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
import io
import sys
import six
import numpy as np
import paddle.fluid.incubate.data_generator as dg


class NumpyRandomInt(object):
    def __init__(self, a, b, buf_size=1000):
        self.idx = 0
        self.buffer = np.random.randint(a, b, buf_size)
        self.a = a
        self.b = b

    def __call__(self):
        if self.idx == len(self.buffer):
            self.buffer = np.random.randint(self.a, self.b, len(self.buffer))
            self.idx = 0

        result = self.buffer[self.idx]
        self.idx += 1
        return result


class Reader(dg.MultiSlotDataGenerator):
    def init(self, config):
        self.window_size = config.get("hyper_parameters.window_size")
        self.neg_num = config.get("hyper_parameters.neg_num")
        self.with_shuffle_batch = config.get(
            "hyper_parameters.with_shuffle_batch")
        self.batch_size = config.get("static_benchmark.batch_size")
        self.random_generator = NumpyRandomInt(1, self.window_size + 1)
        dict_path = config.get("static_benchmark.word_count_dict_path")

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

        self.is_infer = config.get("is_infer", False)
        if self.is_infer:
            self.word_to_id = self.prepare_data(dict_path)

    def BuildWord_IdMap(self, dict_path):
        word_to_id = dict()
        id_to_word = dict()
        with io.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
                id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]
        return word_to_id, id_to_word

    def prepare_data(self, dict_path):
        w2i, _ = self.BuildWord_IdMap(dict_path)
        return w2i

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

    def generate_sample(self, line):
        "Dataset Generator"

        def reader():
            neg_array = self.cs.searchsorted(np.random.sample(self.neg_num))
            id_ = 0
            word_ids = [w for w in line.split()]
            for idx, target_id in enumerate(word_ids):
                context_word_ids = self.get_context_words(word_ids, idx)
                for context_id in context_word_ids:
                    neg_id = [int(str(i)) for i in neg_array]
                    output = [('input_word', [int(target_id)]),
                              ('true_label',
                               [int(context_id)]), ('neg_label', neg_id)]
                    yield output
                    id_ += 1
                    if id_ % self.batch_size == 0:
                        neg_array = self.cs.searchsorted(
                            np.random.sample(self.neg_num))

        return reader

    def dataloader(self, file_list):
        "DataLoader Generator"

        def reader():
            for file in file_list:
                with open(file, 'r') as f:
                    for line in f:
                        neg_array = self.cs.searchsorted(
                            np.random.sample(self.neg_num))
                        id_ = 0
                        word_ids = [w for w in line.split()]
                        for idx, target_id in enumerate(word_ids):
                            context_word_ids = self.get_context_words(word_ids,
                                                                      idx)
                            for context_id in context_word_ids:
                                neg_id = [int(str(i)) for i in neg_array]
                                output = [[int(target_id)], [int(context_id)],
                                          [neg_id]]
                                yield output
                                id_ += 1
                                if id_ % self.batch_size == 0:
                                    neg_array = self.cs.searchsorted(
                                        np.random.sample(self.neg_num))

        def infer_reader():
            for file in file_list:
                with open(file, 'r') as f:
                    for line in f:
                        if ':' in line:
                            pass
                        else:
                            line = self.strip_lines(line.lower(),
                                                    self.word_to_id)
                            line = line.split()
                            yield [self.word_to_id[line[0]]
                                   ], [self.word_to_id[line[1]]], [
                                       self.word_to_id[line[2]]
                                   ], [self.word_to_id[line[3]]], [
                                       self.word_to_id[line[0]],
                                       self.word_to_id[line[1]],
                                       self.word_to_id[line[2]]
                                   ]

        if self.is_infer:
            return infer_reader
        else:
            return reader

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

    def native_to_unicode(self, s):
        if self._is_unicode(s):
            return s
        try:
            return self._to_unicode(s)
        except UnicodeDecodeError:
            res = self._to_unicode(s, ignore_errors=True)
            return res

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


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    utils_path = sys.argv[2]
    sys.path.append(utils_path)
    import utils
    yaml_helper = utils.YamlHelper()
    config = yaml_helper.load_yaml(yaml_path)

    r = Reader()
    r.init(config)
    r.run_from_stdin()
