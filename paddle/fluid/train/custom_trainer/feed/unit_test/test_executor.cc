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

#include <gtest/gtest.h>

#include "paddle/fluid/train/custom_trainer/feed/executor/executor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

TEST(testSimpleExecute, initialize) {
    SimpleExecute execute;
    auto context_ptr = std::make_shared<TrainerContext>();
    YAML::Node config = YAML::Load("[1, 2, 3]");
    ASSERT_NE(0, execute.initialize(config, context_ptr));
    config = YAML::Load("{startup_program: ./data/startup_program, main_program: ./data/main_program}");
    ASSERT_EQ(0, execute.initialize(config, context_ptr));
    config = YAML::Load("{thread_num: 2, startup_program: ./data/startup_program, main_program: ./data/main_program}");
    ASSERT_EQ(0, execute.initialize(config, context_ptr));
}

TEST(testSimpleExecute, run) {
    SimpleExecute execute;
    auto context_ptr = std::make_shared<TrainerContext>();
    auto config = YAML::Load("{thread_num: 2, startup_program: ./data/startup_program, main_program: ./data/main_program}");
    ASSERT_EQ(0, execute.initialize(config, context_ptr));
    ASSERT_EQ(0, execute.run());
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
