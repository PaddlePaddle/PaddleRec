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
"""
The file implements data preprocessing and dataset spilting.
"""

from __future__ import print_function
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil
import pickle
import csv
from collections import defaultdict
import logging
import argparse
import os
import sys

sys.path.append("../../")
from tools.utils.utils_single import load_yaml, load_dy_model_class, get_abs_model, create_data_loader


def _mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


class PreDataset(object):
    def __init__(self, config):
        super(PreDataset, self).__init__()
        self.config = config

        self.field_names = None
        self.target_name = None
        self.field_info = None
        self.idx_to_field_name = None
        self.feature_map = None
        self.train_cnt = 0
        self.test_cnt = 0
        self.sample_cnt = 0
        self.raw_file_dir = self.config.get("runner.raw_file_dir")
        self.raw_filled_file_dir = self.config.get(
            "runner.raw_filled_file_dir")

        self.rebuild_feature_map = self.config.get(
            "runner.rebuild_feature_map")
        self.min_threshold = self.config.get("runner.min_threshold")
        self.feature_map_cache = self.config.get("runner.feature_map_cache")

        self.filled_raw()

        self.init()

    def init(self):
        self._get_field_name()
        self._get_feature_map()
        self._build_split()

    def filled_raw(self):
        "fill raw data with '-1' ,and spilt user, item, contex fields"
        train_path = self.raw_file_dir
        _mkdir_if_not_exist(self.raw_filled_file_dir)
        self.file_object = self.raw_filled_file_dir + '/PreRaw_data.txt'

        file_object_ = open(self.file_object, 'w')
        with open(train_path, "r") as rf:
            n = 0
            m = -1
            for l in tqdm(rf):
                m += 1
                out = []
                values = l.rstrip('\n').split(',')

                fields_values = []
                for i, v in enumerate(values):
                    if v == "":
                        values[i] = "-1"

                fields_values.append(values[0])
                fields_values.append(values[3])
                fields_values.extend(values[16:])
                fields_values.extend(values[11:15])
                fields_values.extend(values[8:11])
                fields_values.extend(values[4:8])
                fields_values.append(values[15])
                fields_values.append(values[2])
                fields_values.append(values[1])

                if m == 0:
                    print(fields_values)
                file_object_.write(','.join(fields_values) + '\n')
        file_object_.close()
        logging.info('All Samples: %s ' % (m))

    def _get_field_name(self):
        self.file_object = self.raw_filled_file_dir + '/PreRaw_data.txt'  ##################
        with open(self.file_object) as csv_file:  # open the input file.
            data_file = csv.reader(csv_file)
            header = next(data_file)  # get the header line.
            self.field_info = {k: v for v, k in enumerate(header)}
            self.idx_to_field_name = {
                idx: name
                for idx, name in enumerate(header)
            }
            self.field_names = header[2:]  # list of feature names.
            self.field_names.append(header[0])
            self.target_name = header[1]  # target name.

    def _get_feature_map(self):
        if not self.rebuild_feature_map and Path(
                self.feature_map_cache).exists():
            with open(self.feature_map_cache, 'rb') as f:
                feature_mapper = pickle.load(f)
        else:
            feature_cnts = defaultdict(lambda: defaultdict(int))
            with open(self.file_object) as f:
                f.readline()
                pbar = tqdm(f, mininterval=1, smoothing=0.1)
                pbar.set_description('Create avazu dataset: counting features')
                for line in pbar:
                    values = line.rstrip('\n').split(',')
                    if len(values) != len(self.field_names) + 1:
                        continue
                    for k, v in self.field_info.items():
                        if k not in ['click']:
                            feature_cnts[k][values[v]] += 1
            feature_mapper = {
                field_name: {
                    feature_name
                    for feature_name, c in cnt.items()
                    if c >= self.min_threshold
                }
                for field_name, cnt in feature_cnts.items()
            }
            feature_mapper['id'] = {
                feature_name
                for feature_name, c in feature_cnts['id'].items()
            }
            feature_mapper = {
                field_name:
                {feature_name: idx
                 for idx, feature_name in enumerate(cnt)}
                for field_name, cnt in feature_mapper.items()
            }

            shutil.rmtree(self.feature_map_cache, ignore_errors=True)
            with open(self.feature_map_cache, 'wb') as f:
                pickle.dump(feature_mapper, f)

        self.feature_map = feature_mapper

    def _build_split(self):
        full_lines = []
        self.data = []

        _mkdir_if_not_exist(self.config.get("runner.train_data_dir"))
        _mkdir_if_not_exist(self.config.get("runner.test_data_dir"))

        train_file = open(
            os.path.join(
                self.config.get("runner.train_data_dir"), 'train_data.txt'),
            'w')
        test_file = open(
            os.path.join(
                self.config.get("runner.test_data_dir"), 'test_data.txt'), 'w')

        features = {}  # dict for all feature columns and target column.

        feature_mapper = self.feature_map
        sample_cnt = 0
        for file in [self.file_object]:
            with open(file, "r") as rf:
                train_cnt = 0
                test_cnt = 0
                rf.readline()
                pbar = tqdm(rf, mininterval=1, smoothing=0.1)
                pbar.set_description(
                    'Split avazu dataset: train_dataset and test_dataset')
                for line in pbar:
                    sample_cnt += 1

                    values = line.rstrip('\n').split(',')

                    if len(values) != len(self.field_names) + 1:
                        continue

                    features = {
                        self.idx_to_field_name[idx]:
                        feature_mapper[self.idx_to_field_name[idx]][value]
                        for idx, value in enumerate(values)
                        if self.idx_to_field_name[idx] != 'click' and value in
                        feature_mapper[self.idx_to_field_name[idx]]
                    }
                    features.update({'target': values[-1]})

                    if "14103000" in values[22]:
                        test_cnt += 1
                        value_n = 0
                        for k, v in features.items():
                            value_n += 1
                            if value_n == len(list(features.values())):
                                test_file.write(str(v) + '\n')
                            else:
                                test_file.write(str(v) + ',')
                    else:
                        train_cnt += 1
                        value_n = 0
                        for k, v in features.items():
                            value_n += 1
                            if value_n == len(list(features.values())):
                                train_file.write(str(v) + '\n')
                            else:
                                train_file.write(str(v) + ',')

            self.train_cnt = train_cnt
            self.test_cnt = test_cnt
            self.sample_cnt = sample_cnt


def main(args):
    config = load_yaml(args.config_yaml)

    logging.info("Starting preprocess dataset ...")
    data = PreDataset(config)
    logging.info("Finished preprocess dataset!")
    train_cnt = data.train_cnt
    test_cnt = data.test_cnt
    samples = data.sample_cnt
    fields = len(data.field_names)

    logging.info('All Samples: %s ' % (samples))
    logging.info('Train Samples: %s ' % (train_cnt))
    logging.info('Test Samples: %s ' % (test_cnt))
    logging.info('Fields: %s ' % (fields))


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(
        description="Parameter of preprocess data")
    parser.add_argument("-m", "--config_yaml", type=str)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)

    args = parser.parse_args()

    main(args)
