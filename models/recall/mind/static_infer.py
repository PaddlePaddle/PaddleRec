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

from __future__ import print_function
import argparse
import time

import os
import warnings
import logging
import paddle
import sys
import numpy as np
import math
__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(
    os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'tools')))

from utils.save_load import save_static_model, load_static_model
from utils.utils_single import load_yaml, load_static_model_class, get_abs_model, create_data_loader, reset_auc

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train static script")
    parser.add_argument("-m", "--config_yaml", type=str)
    parser.add_argument("-top_n", "--top_n", type=int, default=20)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(12345)

    # load config
    config = load_yaml(args.config_yaml)
    config["config_abs_dir"] = args.abs_dir
    # load static model class
    static_model_class = load_static_model_class(config)

    input_data = static_model_class.create_feeds(is_infer=True)
    input_data_names = [data.name for data in input_data]

    fetch_vars = static_model_class.infer_net(input_data)
    logger.info("cpu_num: {}".format(os.getenv("CPU_NUM")))

    use_gpu = config.get("runner.use_gpu", True)
    use_auc = config.get("runner.use_auc", False)
    auc_num = config.get("runner.auc_num", 1)
    test_data_dir = config.get("runner.test_data_dir", None)
    print_interval = config.get("runner.print_interval", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)
    batch_size = config.get("runner.infer_batch_size", None)
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 1))
    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    exe = paddle.static.Executor(place)
    # initialize
    exe.run(paddle.static.default_startup_program())

    test_dataloader = create_data_loader(
        config=config, place=place, mode="test")

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_static_model(
            paddle.static.default_main_program(),
            model_path,
            prefix='rec_static')

        epoch_begin = time.time()
        interval_begin = time.time()

        b = paddle.static.global_scope().find_var("item_emb").get_tensor()
        b = np.array(b)

        import faiss
        if use_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatIP(res, b.shape[-1], flat_config)
            faiss_index.add(b)
        else:
            faiss_index = faiss.IndexFlatIP(b.shape[-1])
            faiss_index.add(b)

        total = 1
        total_recall = 0.0
        total_ndcg = 0.0
        total_hitrate = 0

        for batch_id, batch_data in enumerate(test_dataloader()):
            fetch_batch_var = exe.run(
                program=paddle.static.default_main_program(),
                feed=dict(zip(input_data_names, batch_data[:2])),
                fetch_list=[var for _, var in fetch_vars.items()])

            user_embs = fetch_batch_var[0]
            target_items = np.squeeze(np.array(batch_data[-1]), axis=1)

            if len(user_embs.shape) == 2:
                D, I = faiss_index.search(user_embs, args.top_n)
                for i, iid_list in enumerate(target_items):
                    recall = 0
                    dcg = 0.0
                    item_list = set(I[i])
                    iid_list = list(filter(lambda x: x != 0, list(iid_list)))
                    for no, iid in enumerate(iid_list):
                        if iid in item_list:
                            recall += 1
                            dcg += 1.0 / math.log(no + 2, 2)
                    idcg = 0.0
                    for no in range(recall):
                        idcg += 1.0 / math.log(no + 2, 2)
                    total_recall += recall * 1.0 / len(iid_list)
                    if recall > 0:
                        total_ndcg += dcg / idcg
                        total_hitrate += 1
            else:
                ni = user_embs.shape[1]
                user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
                D, I = faiss_index.search(user_embs, args.top_n)
                for i, iid_list in enumerate(target_items):
                    recall = 0
                    dcg = 0.0
                    item_list_set = set()
                    item_list = list(
                        zip(
                            np.reshape(I[i * ni:(i + 1) * ni], -1),
                            np.reshape(D[i * ni:(i + 1) * ni], -1)))
                    item_list.sort(key=lambda x: x[1], reverse=True)
                    for j in range(len(item_list)):
                        if item_list[j][0] not in item_list_set and item_list[
                                j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            if len(item_list_set) >= args.top_n:
                                break
                    iid_list = list(filter(lambda x: x != 0, list(iid_list)))
                    for no, iid in enumerate(iid_list):
                        if iid == 0:
                            break
                        if iid in item_list_set:
                            recall += 1
                            dcg += 1.0 / math.log(no + 2, 2)
                    idcg = 0.0
                    for no in range(recall):
                        idcg += 1.0 / math.log(no + 2, 2)

                    total_recall += recall * 1.0 / len(iid_list)
                    if recall > 0:
                        total_ndcg += dcg / idcg
                        total_hitrate += 1
            total += target_items.shape[0]

            if batch_id % print_interval == 0:
                recall = total_recall / total
                ndcg = total_ndcg / total
                hitrate = total_hitrate * 1.0 / total
                metric_str = ""
                metric_str += "recall@%d: %.5f, " % (args.top_n, recall)
                metric_str += "ndcg@%d: %.5f, " % (args.top_n, ndcg)
                metric_str += "hitrate@%d: %.5f, " % (args.top_n, hitrate)
                logger.info("epoch: {}, batch_id: {}, ".format(
                    epoch_id, batch_id) + metric_str + "speed: {:.2f} ins/s".
                            format(print_interval * batch_size / (time.time(
                            ) - interval_begin)))

        recall = total_recall / total
        ndcg = total_ndcg / total
        hitrate = total_hitrate * 1.0 / total
        metric_str = ""
        metric_str += "recall@%d: %.5f, " % (args.top_n, recall)
        metric_str += "ndcg@%d: %.5f, " % (args.top_n, ndcg)
        metric_str += "hitrate@%d: %.5f, " % (args.top_n, hitrate)

        logger.info("epoch: {} done, ".format(epoch_id) + metric_str +
                    "epoch time: {:.2f} s".format(time.time() - epoch_begin))


if __name__ == "__main__":
    paddle.enable_static()
    args = parse_args()
    main(args)
