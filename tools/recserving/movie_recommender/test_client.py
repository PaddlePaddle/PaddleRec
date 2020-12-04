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

import sys
import grpc
import proto.um_pb2 as um_pb2
import proto.um_pb2_grpc as um_pb2_grpc
import proto.mm_pb2 as mm_pb2
import proto.mm_pb2_grpc as mm_pb2_grpc
import proto.rank_pb2 as rank_pb2
import proto.rank_pb2_grpc as rank_pb2_grpc
import proto.recall_pb2 as recall_pb2
import proto.recall_pb2_grpc as recall_pb2_grpc
import proto.gr_pb2 as gr_pb2
import proto.gr_pb2_grpc as gr_pb2_grpc
import json
from google.protobuf.json_format import MessageToJson, Parse


def get_ums(uid):
    channel = grpc.insecure_channel('127.0.0.1:8910')
    stub = um_pb2_grpc.UMServiceStub(channel)
    request = um_pb2.UserModelRequest()
    request.user_id = unicode(uid)
    response = stub.um_call(request)
    return response

def get_recall(request):

    channel = grpc.insecure_channel('127.0.0.1:8950')
    stub = recall_pb2_grpc.RecallServiceStub(channel)
    response = stub.recall(request)
    return response

def get_mm(nid_list):
    channel = grpc.insecure_channel('127.0.0.1:8920')
    stub = mm_pb2_grpc.MMServiceStub(channel)
    mm_request = mm_pb2.MMRequest()
    for nid in nid_list:
        mm_request.item_ids.append(str(nid).encode(encoding='utf-8'))
    mm_response = stub.mm_call(mm_request,timeout=10)
    return mm_response

def get_rank(request):
    channel = grpc.insecure_channel('127.0.0.1:8960')
    stub = rank_pb2_grpc.RankServiceStub(channel)
    response = stub.rank_predict(request)
    return response


def get_gr(request):
    channel = grpc.insecure_channel("127.0.0.1:8930")
    stub = gr_pb2_grpc.GRServiceStub(channel)
    response = stub.gr_call(request)
    return response


if __name__ == "__main__":
    if sys.argv[1] == 'gr':
        uid = sys.argv[2]
        req = gr_pb2.GRRequest()
        req.user_id = uid
        print(get_gr(req))
    if sys.argv[1] == 'um':
        uid = sys.argv[2]
        print(get_ums(uid))
    if sys.argv[1] == 'mm':
        nid_list_str= sys.argv[2]
        nid_list = nid_list_str.strip().split(",")
        print(get_mm(nid_list))
    if sys.argv[1] == "recall":
        request = recall_pb2.RecallRequest()
        request.user_info.user_id="1"
        request.user_info.age=3
        request.user_info.job="1"
        request.user_info.gender="M"
        print(get_recall(request))
    if sys.argv[1] == "rank":
        request = rank_pb2.RankRequest()
        request.user_info.user_id="1"
        request.user_info.age=3
        request.user_info.job="1"
        request.user_info.gender="M"
        item_info = request.item_infos.add()
        item_info.movie_id = "1"
        item_info.title="ET"
        item_info.genre="fiction"
        print(get_rank(request))
