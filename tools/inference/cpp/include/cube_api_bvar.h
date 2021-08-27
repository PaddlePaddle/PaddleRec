// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "bvar/bvar.h"
#include "bvar/recorder.h"
#include "bvar/window.h"

namespace rec {
namespace mcube {
bvar::Adder<uint64_t> g_cube_keys_num("cube_keys_num");
bvar::Window<bvar::Adder<uint64_t>> g_cube_keys_num_minute(
    "cube_keys_num_minute", &g_cube_keys_num, 60);
bvar::Adder<uint64_t> g_cube_keys_miss_num("cube_keys_miss_num");
bvar::Window<bvar::Adder<uint64_t>> g_cube_keys_miss_num_minute(
    "cube_keys_miss_num_minute", &g_cube_keys_miss_num, 60);
bvar::IntRecorder g_cube_value_size("cube_value_size");
bvar::Window<bvar::IntRecorder> g_cube_value_size_win(
    "cube_value_size_win", &g_cube_value_size, bvar::FLAGS_bvar_dump_interval);
}  // namespace mcube
}  // namespace rec
