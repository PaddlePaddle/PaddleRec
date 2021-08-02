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

from proto import recall_pb2 
from proto import recall_pb2_grpc 
from proto import user_info_pb2 as user_info_pb2
import redis
# from milvus import Milvus, DataType
from paddle_serving_app.local_predict import LocalPredictor
import numpy as np

import sys
sys.path.append("..")
from milvus_tool.milvus_recall import RecallByMilvus

def hash2(a):
    return hash(a) % 60000000

class RecallServerServicer(object):
    def __init__(self):
        self.uv_client = LocalPredictor()
        self.uv_client.load_model_config("user_vector_model/serving_server_dir") 
        # milvus_host = '127.0.0.1'
        # milvus_port = '19530'
        self.milvus_client = RecallByMilvus()
        self.collection_name = 'demo_films'

    def get_user_vector(self, user_info):
        dic = {"userid": [], "gender": [], "age": [], "occupation": []}
        lod = [0]
        dic["userid"].append(hash2(user_info.user_id))
        dic["gender"].append(hash2(user_info.gender))
        dic["age"].append(hash2(user_info.age))
        dic["occupation"].append(hash2(user_info.job))
        lod.append(1)

        dic["userid.lod"] = lod
        dic["gender.lod"] = lod
        dic["age.lod"] = lod
        dic["occupation.lod"] = lod
        for key in dic:
            dic[key] = np.array(dic[key]).astype(np.int64).reshape(len(dic[key]),1)

        fetch_map = self.uv_client.predict(feed=dic, fetch=["save_infer_model/scale_0.tmp_0"], batch=True)
        return fetch_map["save_infer_model/scale_0.tmp_0"].tolist()[0]

    def recall(self, request, context):
        '''
    message RecallRequest{
        string log_id = 1;
        user_info.UserInfo user_info = 2;
        string recall_type= 3;
        uint32 request_num= 4;
    }

    message RecallResponse{
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
    }
        '''
        recall_res = recall_pb2.RecallResponse()
        user_vector = self.get_user_vector(request.user_info)

        status, results = self.milvus_client.search(collection_name=self.collection_name, vectors=[user_vector], partition_tag="Movie")
        for entities in results:
            if len(entities) == 0:
                recall_res.error.code = 500
                recall_res.error.text = "Recall server get milvus fail. ({})".format(str(request))
                return recall_res
            for topk_film in entities:
                # current_entity = topk_film.entity
                score_pair = recall_res.score_pairs.add()
                score_pair.nid = str(topk_film.id)
                score_pair.score = float(topk_film.distance)
        recall_res.error.code = 200
        return recall_res

class RecallServer(object):
    """
    recall server
    """
    def start_server(self):
        max_workers = 40
        concurrency = 40
        port = 8950
        
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        servicer = RecallServerServicer()
        recall_pb2_grpc.add_RecallServiceServicer_to_server(servicer, server)
        server.add_insecure_port('[::]:{}'.format(port))
        server.start()
        server.wait_for_termination()

if __name__ == "__main__":
    recall = RecallServer()
    recall.start_server()
