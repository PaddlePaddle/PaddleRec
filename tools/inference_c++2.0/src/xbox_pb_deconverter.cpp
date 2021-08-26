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

#include <stdio.h>
#include "xbox_pb_deconverter.h"

DECLARE_bool(debug);
DECLARE_string(cube_result_file);

namespace paddleRecInfer {
namespace xbox_pb_converter {
    
void XboxPbDeconverter::Deconvert(uint64_t& key, std::string& value) {
    Init();
    PaddleRec::infer::MioMFResult mio_pb;
    mio_pb.ParseFromString(value);
    if (mio_pb.has_show() && mio_pb.show() != 0) {
            show = mio_pb.show() / showRatio;
        }
    if (mio_pb.has_click() && mio_pb.click() != 0) {
        clk = mio_pb.click() / clkRatio;
    }
    if (mio_pb.has_pred() && mio_pb.pred() != 0) {
        pred = mio_pb.pred();
    }
    if (mio_pb.has_weight_lr_person() && mio_pb.weight_lr_person() != 0) {
        lr = 1.0 * mio_pb.weight_lr_person() / lrRatio;
    }
    size_t mf_count = mio_pb.weight_mf_size();
    if (mf_count != 0) {
        for (size_t i = 0; i < mf_count; ++i) {
            mf.push_back(1.0 * mio_pb.weight_mf(i) / mfRatio);
        }
    }
    if (FLAGS_debug) {
        FILE* fout = fopen(const_cast<char*>(FLAGS_cube_result_file.c_str()), "a+");
        CHECK(fprintf(fout, "%lu\t", key) > 0);
        CHECK(fprintf(fout, "%f,", lr) > 0);
        for (uint i = 0; i < mf.size(); i++) {
            if (i == (mf.size() - 1)) {
                CHECK(fprintf(fout, "%f", mf[i]) > 0);
            } else {
                CHECK(fprintf(fout, "%f,", mf[i]) > 0);
            }
        }
        CHECK(fprintf(fout, "\n"));
        CHECK(fflush(fout) == 0);
        fclose(fout);
    }
};

}
}
