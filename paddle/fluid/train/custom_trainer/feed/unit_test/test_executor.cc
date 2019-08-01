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
#include <fstream>
#include <gtest/gtest.h>

#include "paddle/fluid/train/custom_trainer/feed/executor/executor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/io/fs.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

const char test_data_dir[] = "test_data";
const char main_program_path[] = "test_data/main_program";
const char startup_program_path[] = "test_data/startup_program";

class SimpleExecutorTest : public testing::Test
{
public:
    static void SetUpTestCase()
    {
        ::paddle::framework::localfs_mkdir(test_data_dir);

        {
            std::unique_ptr<paddle::framework::ProgramDesc> startup_program(
                new paddle::framework::ProgramDesc());
            std::ofstream fout(startup_program_path, std::ios::out | std::ios::binary);
            ASSERT_TRUE(fout);
            fout << startup_program->Proto()->SerializeAsString();
            fout.close();
        }

        {
            std::unique_ptr<paddle::framework::ProgramDesc> main_program(
                new paddle::framework::ProgramDesc());

            auto load_block = main_program->MutableBlock(0);
            framework::OpDesc* op = load_block->AppendOp();
            op->SetType("mean");
            op->SetInput("X", {"x"});
            op->SetOutput("Out", {"mean"});
            op->CheckAttrs();
            std::ofstream fout(main_program_path, std::ios::out | std::ios::binary);
            ASSERT_TRUE(fout);
            fout << main_program->Proto()->SerializeAsString();
            fout.close();
        }
    }

    static void TearDownTestCase()
    {
        ::paddle::framework::localfs_remove(test_data_dir);
    }

    virtual void SetUp()
    {
        context_ptr.reset(new TrainerContext());
    }

    virtual void TearDown()
    {
        context_ptr = nullptr;
    }

    std::shared_ptr<TrainerContext> context_ptr;
};

TEST_F(SimpleExecutorTest, initialize) {
    std::unique_ptr<Executor> executor(CREATE_CLASS(Executor, "SimpleExecutor"));
    YAML::Node config = YAML::Load("[1, 2, 3]");
    ASSERT_NE(0, executor->initialize(config, context_ptr));
    config = YAML::Load(std::string() + "{startup_program: " + startup_program_path + ", main_program: " + main_program_path + "}");
    ASSERT_EQ(0, executor->initialize(config, context_ptr));
    config = YAML::Load(std::string() + "{thread_num: 2, startup_program: " + startup_program_path + ", main_program: " + main_program_path + "}");
    ASSERT_EQ(0, executor->initialize(config, context_ptr));
}

TEST_F(SimpleExecutorTest, run) {
    std::unique_ptr<Executor> executor(CREATE_CLASS(Executor, "SimpleExecutor"));

    auto config = YAML::Load(std::string() + "{thread_num: 2, startup_program: " + startup_program_path + ", main_program: " + main_program_path + "}");
    ASSERT_EQ(0, executor->initialize(config, context_ptr));
    
	auto x_var = executor->mutable_var<::paddle::framework::LoDTensor>("x");
    executor->mutable_var<::paddle::framework::LoDTensor>("mean");
    ASSERT_NE(nullptr, x_var);

    int x_len = 10;
	x_var->Resize({1, x_len});
	auto x_data = x_var->mutable_data<float>(context_ptr->cpu_place);
    std::cout << "x: ";
    for (int i = 0; i < x_len; ++i) {
        x_data[i] = i;
        std::cout << i << " ";
    }
    std::cout << std::endl;

    ASSERT_EQ(0, executor->run());

	auto mean_var = executor->var<::paddle::framework::LoDTensor>("mean");
    auto mean = mean_var.data<float>()[0];
    std::cout << "mean: " << mean << std::endl;
    ASSERT_NEAR(4.5, mean, 1e-9);
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
