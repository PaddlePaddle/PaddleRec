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


def serialize_example(dense_feature, sparse_feature, label):
    """
  Creates a tf.train.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'dense_feature': _float_feature(dense_feature),
        'sparse_feature': _bytes_feature(sparse_feature),
        'label': _int64_feature(label),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(
        feature=feature))
    return example_proto.SerializeToString()


if __name__ == '__main__':

    output_file = 'wd.tfrecord'
    input_file = './part-0'

    # setting for this dataset only
    continuous_range_ = range(1, 14)
    categorical_range_ = range(14, 40)

    writer = tf.io.TFRecordWriter(output_file)

    print("begin write file")
    with open(input_file, 'r') as f:
        for line in f:
            features = line.rstrip('\n').split('\t')
            dense_feature = []
            sparse_feature = []
            for idx in continuous_range_:
                if features[idx] == "":
                    dense_feature.append(0.0)
                else:
                    dense_feature.append(float(features[idx]))
            for idx in categorical_range_:
                sparse_feature.append(features[idx].encode('utf8'))
            label = int(features[0])

            example = serialize_example(dense_feature, sparse_feature, label)
            writer.write(example)

    writer.flush()
    writer.close()
    print("write file done")
