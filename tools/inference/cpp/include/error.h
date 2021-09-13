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

namespace rec {
namespace mcube {

struct CubeError {
  enum Code {
    E_OK = 0,
    E_NO_SUCH_KEY = -1,
    E_SEEK_FAILED = -2,
    E_ALL_SEEK_FAILED = -3,
  };  // enum Code

  static const char* error_msg(Code code);
};  // struct CubeError

}  // namespace mcube
}  // namespace rec
