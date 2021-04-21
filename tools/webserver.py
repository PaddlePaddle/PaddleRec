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

import sys
import numpy as np
import argparse
import paddle
import os
from paddle_serving_client import Client
from paddle_serving_app.local_predict import LocalPredictor
if sys.argv[1] == 'gpu':
    from paddle_serving_server_gpu.web_service import WebService
elif sys.argv[1] == 'cpu':
    from paddle_serving_server.web_service import WebService


class RecService(WebService):
    def preprocess(self, feed=[], fetch=[]):
        feed_dict = {}
        feed = feed[0]
        for key in feed.keys():
            feed_dict[key] = np.array(feed[key])
        return feed_dict, fetch, True

    def postprocess(self, feed=[], fetch=[], fetch_map=None):
        print(fetch)
        print(fetch_map)
        fetch_map = {x: fetch_map[x].tolist() for x in fetch_map.keys()}
        return fetch_map


rec_service = RecService(name="rec")
#rec_service.setup_profile(30)
rec_service.load_model_config("serving_server")
rec_service.prepare_server(workdir="workdir", port=int(sys.argv[2]))
if sys.argv[1] == 'gpu':
    rec_service.set_gpus("0")
    rec_service.run_debugger_service(gpu=True)
elif sys.argv[1] == 'cpu':
    rec_service.run_debugger_service()
rec_service.run_web_service()
