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
import paddle.fluid.incubate.data_generator as dg
import config


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


class Generator(dg.MultiSlotDataGenerator):
    """
    DacDataset: inheritance MultiSlotDataGeneratior, Implement data reading
    Help document: http://wiki.baidu.com/pages/viewpage.action?pageId=728820675
    """

    def init(self):
        self.window_size = config.window_size
        self.neg_num = config.neg_num
        self.with_shuffle_batch = config.with_shuffle_batch
        self.random_generator = NumpyRandomInt(1, self.window_size + 1)
        self.batch_size = config.batch_size

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

    def generate_sample(self, line):
        "Dataset Generator"

        def reader():
            word_ids = [w for w in line.split()]
            for idx, target_id in enumerate(word_ids):
                context_word_ids = self.get_context_words(word_ids, idx)
                for context_id in context_word_ids:
                    output = [('input_word', [int(target_id)]),
                              ('true_label', [int(context_id)])]
                    if config.with_shuffle_batch:
                        yield output
                    else:
                        neg_array = self.cs.searchsorted(
                            np.random.sample(self.neg_num))
                        output += [('neg_label',
                                    [int(str(i)) for i in neg_array])]
                        yield output

        return reader

    def dataloader(self, file_list):
        "DataLoader Generator"

        def reader():
            for file in file_list:
                with open(file, 'r') as f:
                    for line in f:
                        word_ids = [w for w in line.split()]
                        for idx, target_id in enumerate(word_ids):
                            context_word_ids = self.get_context_words(word_ids,
                                                                      idx)
                            for context_id in context_word_ids:
                                output = [[int(target_id)], [int(context_id)]]
                                if config.with_shuffle_batch:
                                    yield output
                                else:
                                    neg_array = self.cs.searchsorted(
                                        np.random.sample(self.neg_num))
                                    output += [[
                                        int(str(i)) for i in neg_array
                                    ]]
                                    yield output

        return reader


if __name__ == "__main__":
    d = Generator()
    d.init()
    d.run_from_stdin()
