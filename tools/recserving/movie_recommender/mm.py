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

from proto import mm_pb2 
from proto import mm_pb2_grpc 
from proto import item_info_pb2 as item_info_pb2
import redis
import json

class MMServerServicer(object):
    def __init__(self):
        self.redis_cli = redis.StrictRedis(host="127.0.0.1", port="6379")

    def mm_call(self, request, context):
        mm_res = mm_pb2.MMResponse()
        item_ids = request.item_ids;
        for item_id in item_ids:
            redis_res = self.redis_cli.get("{}##movie_info".format(item_id))
            if redis_res is None:
                mm_res.error.code = 500
                mm_res.error.text = "MM server get item_info from redis fail. ({})".format(str(request))
                return um_res
                #raise ValueError("MM server get user_info from redis fail. ({})".format(str(request)))
            mm_info = json.loads(redis_res)
            item_info = mm_res.item_infos.add()
            if "movie_id" not in mm_info:
                raise ValueError("not get movie from mm")
            item_info.movie_id = mm_info["movie_id"]
            item_info.title = mm_info["title"]
            item_info.genre = ', '.join(mm_info["genre"])
        mm_res.error.code = 200
        return mm_res

class MMServer(object):
    """
    mm server
    """
    def start_server(self):
        max_workers = 40
        concurrency = 40
        port = 8920
        
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        servicer = MMServerServicer()
        mm_pb2_grpc.add_MMServiceServicer_to_server(servicer, server)
        server.add_insecure_port('[::]:{}'.format(port))
        server.start()
        server.wait_for_termination()

if __name__ == "__main__":
    um = MMServer()
    um.start_server()
