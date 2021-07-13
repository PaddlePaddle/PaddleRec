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

# !/bin/env python

from __future__ import unicode_literals

from concurrent import futures

import grpc

from proto import rank_pb2 
from proto import rank_pb2_grpc 
from proto import user_info_pb2 as user_info_pb2
import redis
import numpy as np
from paddle_serving_app.local_predict import LocalPredictor
def hash2(a):
    return hash(a) % 60000000

class RankServerServicer(object):
    def __init__(self):
        self.ctr_client = LocalPredictor()
        self.ctr_client.load_model_config("rank_model")

    def process_feed_dict(self, user_info, item_infos):
        #" userid gender age occupation | movieid title genres"
        dic = {"userid": [], "gender": [], "age": [], "occupation": [], "movieid": [], "title": [], "genres": []}
        batch_size = len(item_infos)
        lod = [0]
        for i, item_info in enumerate(item_infos):
            dic["movieid"].append(hash2(item_info.movie_id))
            dic["title"].append(hash2(item_info.title))
            dic["genres"].append(hash2(item_info.genre))
            dic["userid"].append(hash2(user_info.user_id))
            dic["gender"].append(hash2(user_info.gender))
            dic["age"].append(hash2(user_info.age))
            dic["occupation"].append(hash2(user_info.job))
            lod.append(i+1)

        dic["movieid.lod"] = lod
        dic["title.lod"] = lod
        dic["genres.lod"] = lod
        dic["userid.lod"] = lod
        dic["gender.lod"] = lod
        dic["age.lod"] = lod
        dic["occupation.lod"] = lod
        for key in dic:
            dic[key] = np.array(dic[key]).astype(np.int64).reshape(len(dic[key]),1)

        return dic

    def rank_predict(self, request, context):
        '''
        message RankRequest {
          string log_id = 1;
            user_info.UserInfo user_info = 2;
            repeated item_info.ItemInfo item_infos = 3;
        }

        message RankResponse {
            message Error {
                uint32 code = 1;
                string text = 2;
            }
            message ScorePair {
                string nid = 1;
                float score = 2;
            };
            Error error = 1;
            repeated ScorePair score_pairs = 2;
        };
        '''
        batch_size = len(request.item_infos)
        dic = self.process_feed_dict(request.user_info, request.item_infos)
        fetch_map = self.ctr_client.predict(feed=dic, fetch=["save_infer_model/scale_0.tmp_0"], batch=True)
        response = rank_pb2.RankResponse()
        
        #raise ValueError("UM server get user_info from redis fail. ({})".format(str(request)))
        response.error.code = 200

        for i in range(batch_size):
            score_pair = response.score_pairs.add()
            score_pair.nid = request.item_infos[i].movie_id
            score_pair.score = fetch_map["save_infer_model/scale_0.tmp_0"][i][0]
        response.score_pairs.sort(reverse=True, key = lambda item: item.score)
        return response

class RankServer(object):
    """
    rank server
    """
    def start_server(self):
        max_workers = 40
        concurrency = 40
        port = 8960
        
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        servicer = RankServerServicer()
        rank_pb2_grpc.add_RankServiceServicer_to_server(servicer, server)
        server.add_insecure_port('[::]:{}'.format(port))
        server.start()
        server.wait_for_termination()

if __name__ == "__main__":
    rank = RankServer()
    rank.start_server()
