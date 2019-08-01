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

#include <iostream>
#include <gtest/gtest.h>

#include "paddle/fluid/train/custom_trainer/feed/executor/executor.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

TEST(testSimpleExecutor, initialize) {
    SimpleExecutor execute;
    auto context_ptr = std::make_shared<TrainerContext>();
    YAML::Node config = YAML::Load("[1, 2, 3]");
    ASSERT_NE(0, execute.initialize(config, context_ptr));
    config = YAML::Load("{startup_program: ./data/startup_program, main_program: ./data/main_program}");
    ASSERT_EQ(0, execute.initialize(config, context_ptr));
    config = YAML::Load("{thread_num: 2, startup_program: ./data/startup_program, main_program: ./data/main_program}");
    ASSERT_EQ(0, execute.initialize(config, context_ptr));
}

float uniform(float min, float max) {
    float result = (float)rand() / RAND_MAX;
    return min + result * (max - min);
}

void next_batch(int batch_size, const paddle::platform::Place& place, paddle::framework::LoDTensor* x_tensor, paddle::framework::LoDTensor* y_tensor) {
	
	x_tensor->Resize({batch_size, 2});
	auto x_data = x_tensor->mutable_data<float>(place);

	y_tensor->Resize({batch_size, 1});
	auto y_data = y_tensor->mutable_data<float>(place);

    for (int i = 0; i < batch_size; ++i) {
        x_data[i * 2] = uniform(-2, 2);
        x_data[i * 2 + 1] = uniform(-2, 2);
        float dis = x_data[i * 2] * x_data[i * 2] + x_data[i * 2 + 1] * x_data[i * 2 + 1];
        y_data[i] = dis < 1.0 ? 1.0 : 0.0;
    }
}

TEST(testSimpleExecutor, run) {
    SimpleExecutor execute;
    auto context_ptr = std::make_shared<TrainerContext>();
    auto config = YAML::Load("{thread_num: 2, startup_program: ./data/startup_program, main_program: ./data/main_program}");
    ASSERT_EQ(0, execute.initialize(config, context_ptr));

    
	auto x_var = execute.mutable_var<::paddle::framework::LoDTensor>("x");
	auto y_var = execute.mutable_var<::paddle::framework::LoDTensor>("y");
    ASSERT_NE(nullptr, x_var);
    ASSERT_NE(nullptr, y_var);

    next_batch(1024, context_ptr->cpu_place, x_var, y_var);

    ASSERT_EQ(0, execute.run());

	auto loss_var = execute.var<::paddle::framework::LoDTensor>("loss");
    auto loss = loss_var.data<float>()[0];
    std::cout << "loss: " << loss << std::endl;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
