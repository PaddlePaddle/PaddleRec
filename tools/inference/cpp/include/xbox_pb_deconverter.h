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

#ifndef BAIDU_PADDLEPADDLE_PADDLEREC_XBOX_PB_DECONVERTER_H
#define BAIDU_PADDLEPADDLE_PADDLEREC_XBOX_PB_DECONVERTER_H

#include <string>
#include <vector>
#include "glog/logging.h"
#include "miomf_result.pb.h"

namespace paddleRecInfer {
namespace xbox_pb_converter {

class XboxPbDeconverter {
public:
    XboxPbDeconverter() {
    }
    ~XboxPbDeconverter() {
    }
    void Init()
    {
        showRatio = 1;
        clkRatio = 8;
        lrRatio = 1024;
        mfRatio = 1024;
    }
    void Deconvert(uint64_t& key, std::string& value);

private:
    uint showRatio;
    uint clkRatio;
    uint lrRatio;
    uint mfRatio;

public:
    uint64_t show;
    uint64_t clk;
    float pred;
    float lr;
    std::vector<float> mf;
};

}
}
#endif //BAIDU_FCR_MODEL_ABACUS_CONVERTER_XBOX_PB_CONVERTER_INCLUDE_XBOX_PB_DECONVERTER_H
