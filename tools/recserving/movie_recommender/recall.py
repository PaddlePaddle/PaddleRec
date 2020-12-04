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

class RecallServerServicer(object):
    def __init__(self):
        self.redis_cli = redis.StrictRedis(host="127.0.0.1", port="6379")

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
        user_id = request.user_info.user_id;
        redis_res = self.redis_cli.lrange("{}##recall".format(user_id),0,200)
        if redis_res is None:
            recall_res.error.code = 500
            recall_res.error.text = "Recall server get user_info from redis fail. ({})".format(str(request))
            return recall_res
            #raise ValueError("UM server get user_info from redis fail. ({})".format(str(request)))
        recall_res.error.code = 200
        ## FIX HERE
        for item in redis_res:
            item_id, score = item.split("#")[0], item.split("#")[1]
            score_pair = recall_res.score_pairs.add()
            score_pair.nid = item_id
            score_pair.score = float(score)
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
