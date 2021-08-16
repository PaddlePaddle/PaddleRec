# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed.fleet as fleet
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    value = value if isinstance(value, list) else [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    value = value if isinstance(value, list) else [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    value = value if isinstance(value, list) else [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class WideDeepDatasetReader(fleet.MultiSlotDataGenerator):
    def line_process(self, line):
        dense = line["dense_feature"].numpy()
        sparse = line["sparse_feature"].numpy()
        label = [int(line["label"].numpy())]

        dense_array = (dense - cont_min_) / cont_diff_
        dense_feature = [float(i) for i in dense_array]
        sparse_feature = [[hash(i) % hash_dim_] for i in sparse]

        return [dense_feature] + sparse_feature + [label]

    def generate_sample(self, line):
        def wd_reader():
            for line in tf_dataset:

                input_data = self.line_process(line)

                feature_name = ["dense_input"]
                for idx in categorical_range_:
                    feature_name.append("C" + str(idx - 13))
                feature_name.append("label")
                yield zip(feature_name, input_data)

        return wd_reader


if __name__ == "__main__":
    my_data_generator = WideDeepDatasetReader()
    #my_data_generator.set_batch(16)

    filename = 'utils/wd.tfrecord'
    filelists = [filename]
    raw_dataset = tf.data.TFRecordDataset(filelists)

    feature_description = {
        'dense_feature': tf.io.FixedLenFeature(
            [13], tf.float32, default_value=[0.0] * 13),
        'sparse_feature': tf.io.FixedLenFeature(
            [26], tf.string, default_value=[''] * 26),
        'label': tf.io.FixedLenFeature(
            [], tf.int64, default_value=0),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    tf_dataset = raw_dataset.map(_parse_function)

    my_data_generator.run_from_memory()

    # tf will not exit
    exit()
