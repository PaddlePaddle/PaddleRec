#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or aaseed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/bin/env python

import sys
import grpc
import proto.um_pb2 as um_pb2
import proto.um_pb2_grpc as um_pb2_grpc
import proto.cm_pb2 as cm_pb2
import proto.cm_pb2_grpc as cm_pb2_grpc
import proto.rank_pb2 as rank_pb2
import proto.rank_pb2_grpc as rank_pb2_grpc
import proto.recall_pb2 as recall_pb2
import proto.recall_pb2_grpc as recall_pb2_grpc
import proto.as_pb2 as as_pb2
import proto.as_pb2_grpc as as_pb2_grpc
import json
from google.protobuf.json_format import MessageToJson, Parse


def get_ums(uid):
    channel = grpc.insecure_channel('127.0.0.1:8910')
    stub = um_pb2_grpc.UMServiceStub(channel)
    request = um_pb2.UserModelRequest()
    request.user_id = str(uid)
    response = stub.um_call(request)
    return response

def get_recall(request):
    channel = grpc.insecure_channel('127.0.0.1:8950')
    stub = recall_pb2_grpc.RecallServiceStub(channel)
    response = stub.recall(request)
    return response

def get_cm(nid_list):
    channel = grpc.insecure_channel('127.0.0.1:8920')
    stub = cm_pb2_grpc.CMServiceStub(channel)
    cm_request = cm_pb2.CMRequest()
    for nid in nid_list:
        cm_request.item_ids.append(str(nid).encode(encoding='utf-8'))
    cm_response = stub.cm_call(cm_request,timeout=10)
    return cm_response

def get_rank(request):
    channel = grpc.insecure_channel('127.0.0.1:8960')
    stub = rank_pb2_grpc.RankServiceStub(channel)
    response = stub.rank_predict(request)
    return response


def get_as(request):
    channel = grpc.insecure_channel("127.0.0.1:8930")
    stub = as_pb2_grpc.ASServiceStub(channel)
    response = stub.as_call(request)
    return response


if __name__ == "__main__":
    if sys.argv[1] == 'as':
        req = as_pb2.ASRequest()
        if len(sys.argv) == 3:
            uid = sys.argv[2]
            req.user_id = uid
        else:
            gender = sys.argv[2]
            age = int(sys.argv[3])
            job = sys.argv[4]
            req.user_info.user_id, req.user_info.gender, req.user_info.age, req.user_info.job = "0", gender, age, job 
        print(get_as(req))
    if sys.argv[1] == 'um':
        uid = sys.argv[2]
        print(get_ums(uid))
    if sys.argv[1] == 'cm':
        nid_list_str= sys.argv[2]
        nid_list = nid_list_str.strip().split(",")
        print(get_cm(nid_list))
    if sys.argv[1] == "recall":
        uid = sys.argv[2]
        user_info = get_ums(uid).user_info
        request = recall_pb2.RecallRequest()
        request.user_info.CopyFrom(user_info)
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
