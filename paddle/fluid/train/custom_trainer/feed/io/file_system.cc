/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"
#include <string>

namespace paddle {
namespace custom_trainer {
namespace feed {

std::string FileSystem::path_join(const std::string& dir, const std::string& path) {
    if (dir.empty()) {
        return path;
    }
    if (dir.back() == '/') {
        return dir + path;
    }
    return dir + '/' + path;
}

std::pair<std::string, std::string> FileSystem::path_split(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return {".", path};
    }
    return {path.substr(0, pos), path.substr(pos + 1)};
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
