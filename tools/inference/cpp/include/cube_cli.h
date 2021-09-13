// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BAIDU_PADDLEPADDLE_PADDLEREC_CUBE_CLI_H
#define BAIDU_PADDLEPADDLE_PADDLEREC_CUBE_CLI_H

#include <unordered_map>
#include <vector>
#include <set>

namespace rec {
    namespace mcube {
        int run_m(std::set<uint64_t>& keys, std::unordered_map<uint64_t, std::vector<float>>& queryResult);
    }
}

#endif
