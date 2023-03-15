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
from numba import jit

__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(
    os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'tools')))

from utils.save_load import save_model, load_model
from utils.utils_single import load_yaml, get_abs_model, create_data_loader, reset_auc, load_dy_model_class

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PaddleRec train static script")
    parser.add_argument("-m", "--config_yaml", type=str, default="config.yaml")
    parser.add_argument("-o", "--opt", nargs='*', type=str)
    parser.add_argument("-top_n", "--top_n", type=int, default=10)
    args = parser.parse_args()
    args.abs_dir = os.path.dirname(os.path.abspath(args.config_yaml))
    args.config_yaml = get_abs_model(args.config_yaml)
    return args


def main(args):
    paddle.seed(163)

    # load config
    config = load_yaml(args.config_yaml)
    config["config_abs_dir"] = args.abs_dir
    # load static model class
    dy_model_class = load_dy_model_class(config)

    use_gpu = config.get("runner.use_gpu", True)
    test_data_dir = config.get("runner.test_data_dir", None)
    print_interval = config.get("runner.print_interval", None)
    model_load_path = config.get("runner.infer_load_path", "model_output")
    start_epoch = config.get("runner.infer_start_epoch", 0)
    end_epoch = config.get("runner.infer_end_epoch", 10)
    batch_size = config.get("runner.infer_batch_size", None)
    top_k = config.get("runner.top_k", 10)
    os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 1))

    logger.info("**************common.configs**********")
    logger.info(
        "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".
        format(use_gpu, test_data_dir, start_epoch, end_epoch, print_interval,
               model_load_path))
    logger.info("**************common.configs**********")

    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    dy_model = dy_model_class.create_model(config)
    test_dataloader = create_data_loader(
        config=config, place=place, mode="test")

    logger.info("read data")

    epoch_begin = time.time()
    interval_begin = time.time()

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, dy_model)

        recList = {}
        data = dy_model.data
        num_items = dy_model.num_items
        final_user_embeddings, final_item_embeddings = dy_model.infer_embedding(
        )

        def _predict_one(u, i_embeddings, u_embeddings):
            # (1, emb_size)
            user_embedding = paddle.nn.functional.embedding(
                x=u, weight=u_embeddings)
            # (num_item, emb_size) * (emb_size, 1) -> (num_item, 1)
            candidates = paddle.matmul(
                i_embeddings, paddle.transpose(
                    user_embedding, perm=[1, 0]))
            return candidates

        for user in data.testSet_u:
            if data.containsUser(user):
                candidates = paddle.squeeze(
                    _predict_one(
                        paddle.to_tensor([data.getUserId(user)]),
                        final_item_embeddings, final_user_embeddings)).numpy()
            else:
                candidates = [data.globalMean] * num_items

            ratedList, rating_list = data.userRated(user)
            for item in ratedList:
                candidates[data.item[item]] = 0.0
            ids, scores = find_k_largest(top_k, candidates)
            item_names = [data.id2item[iid] for iid in ids]
            recList[user] = list(zip(item_names, scores))

        measure = Measure.rankingMeasure(data.testSet_u, recList, [top_k])

        logger.info("\t".join(measure))


class Measure(object):
    def __init__(self):
        pass

    @staticmethod
    def ratingMeasure(res):
        measure = []
        mae = Measure.MAE(res)
        measure.append('MAE:' + str(mae) + '\n')
        rmse = Measure.RMSE(res)
        measure.append('RMSE:' + str(rmse) + '\n')
        return measure

    @staticmethod
    def hits(origin, res):
        hitCount = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hitCount[user] = len(set(items).intersection(set(predicted)))
        return hitCount

    @staticmethod
    def rankingMeasure(origin, res, N):
        measure = []
        for n in N:
            predicted = {}
            for user in res:
                predicted[user] = res[user][:n]
            indicators = []
            if len(origin) != len(predicted):
                print(
                    'The Lengths of test set and predicted set are not match!')
                exit(-1)
            hits = Measure.hits(origin, predicted)
            prec = Measure.precision(hits, n)
            indicators.append('Precision:' + str(prec) + '\n')
            recall = Measure.recall(hits, origin)
            indicators.append('Recall:' + str(recall) + '\n')
            F1 = Measure.F1(prec, recall)
            indicators.append('F1:' + str(F1) + '\n')
            # MAP = Measure.MAP(origin, predicted, n)
            # indicators.append('MAP:' + str(MAP) + '\n')
            NDCG = Measure.NDCG(origin, predicted, n)
            indicators.append('NDCG:' + str(NDCG) + '\n')
            # AUC = Measure.AUC(origin,res,rawRes)
            # measure.append('AUC:' + str(AUC) + '\n')
            measure.append('Top ' + str(n) + '\n')
            measure += indicators
        return measure

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    # @staticmethod
    # def MAP(origin, res, N):
    #     sum_prec = 0
    #     for user in res:
    #         hits = 0
    #         precision = 0
    #         for n, item in enumerate(res[user]):
    #             if item[0] in origin[user]:
    #                 hits += 1
    #                 precision += hits / (n + 1.0)
    #         sum_prec += precision / min(len(origin[user]), N)
    #     return sum_prec / len(res)

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            # 1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)

    # @staticmethod
    # def AUC(origin, res, rawRes):
    #
    #     from random import choice
    #     sum_AUC = 0
    #     for user in origin:
    #         count = 0
    #         larger = 0
    #         itemList = rawRes[user].keys()
    #         for item in origin[user]:
    #             item2 = choice(itemList)
    #             count += 1
    #             try:
    #                 if rawRes[user][item] > rawRes[user][item2]:
    #                     larger += 1
    #             except KeyError:
    #                 count -= 1
    #         if count:
    #             sum_AUC += float(larger) / count
    #
    #     return float(sum_AUC) / len(origin)

    @staticmethod
    def recall(hits, origin):
        recallList = [hits[user] / len(origin[user]) for user in hits]
        recall = sum(recallList) / len(recallList)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3])
            count += 1
        if count == 0:
            return error
        return error / count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count == 0:
            return error
        return math.sqrt(error / count)


def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid, score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids, k_largest_scores


if __name__ == "__main__":
    args = parse_args()
    main(args)
