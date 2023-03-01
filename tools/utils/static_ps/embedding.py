# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""GPU Parameter Server
"""
import paddle
import paddle.fluid.core as core
from pgl.utils.logger import log
from place import get_cuda_places


class DistEmbedding(object):
    """ Setting the Embedding for the parameter server

    Args:

        slots: a list of int represents the slot key

        embedding_size: the output size of the embedding.
    """

    def __init__(self, slots, embedding_size, slot_num_for_pull_feature):
        self.parameter_server = core.PSGPU()
        self.parameter_server.set_slot_num_for_pull_feature(
            slot_num_for_pull_feature)
        self.parameter_server.set_slot_vector(slots)
        self.parameter_server.init_gpu_ps(get_cuda_places())
        self.parameter_server.set_slot_dim_vector([embedding_size] *
                                                  len(slots))

    def finalize(self):
        """finalize"""
        self.parameter_server.finalize()

    def begin_pass(self):
        """begin pass"""
        self.parameter_server.begin_pass()

    def end_pass(self):
        """end pass"""
        self.parameter_server.end_pass()

    def dump_to_mem(self):
        """dump to mem"""
        self.parameter_server.dump_to_mem()

    def set_infer_mode(self, set_flag=False):
        """set infer mode"""
        self.parameter_server.set_mode(set_flag)

    def __del__(self):
        self.finalize()
