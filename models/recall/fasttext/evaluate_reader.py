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

import six

from paddlerec.core.reader import ReaderBase
from paddlerec.core.utils import envs


class Reader(ReaderBase):
    def init(self):
        dict_path = envs.get_global_env(
            "dataset.dataset_infer.word_id_dict_path")
        self.min_n = envs.get_global_env("hyper_parameters.min_n")
        self.max_n = envs.get_global_env("hyper_parameters.max_n")
        self.word_to_id = dict()
        self.id_to_word = dict()
        with io.open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
                self.id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]
        self.dict_size = len(self.word_to_id)

    def computeSubwords(self, word):
        ngrams = set()
        for i in range(len(word) - self.min_n + 1):
            for j in range(self.min_n, self.max_n + 1):
                end = min(len(word), i + j)
                ngrams.add("".join(word[i:end]))
        return list(ngrams)

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
            "<" + word + ">"
            if "<" + word + ">" in original_vocab else u"<UNK>"
            for word in line.split()
        ])

    def generate_sample(self, line):
        def reader():
            if ':' in line:
                pass
            features = self.strip_lines(line.lower(), self.word_to_id)
            features = features.split()
            inputs = []
            for item in features:
                if item == "<UNK>":
                    inputs.append([self.word_to_id[item]])
                else:
                    ngrams = self.computeSubwords(item)
                    res = []
                    res.append(self.word_to_id[item])
                    for _ in ngrams:
                        res.append(self.word_to_id[_])
                    inputs.append(res)
            yield [('analogy_a', inputs[0]), ('analogy_b', inputs[1]),
                   ('analogy_c', inputs[2]), ('analogy_d', inputs[3][0:1])]

        return reader
