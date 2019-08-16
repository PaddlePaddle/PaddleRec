#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

#include "paddle/fluid/train/custom_trainer/feed/executor/executor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"
#include "paddle/fluid/train/custom_trainer/feed/io/shell.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

namespace {
const char test_data_dir[] = "test_data";
const char main_program_path[] = "test_data/main_program";
const char startup_program_path[] = "test_data/startup_program";
}

class SimpleExecutorTest : public testing::Test
{
public:
    static void SetUpTestCase()
    {
        std::unique_ptr<FileSystem> fs(CREATE_CLASS(FileSystem, "LocalFileSystem"));
        fs->mkdir(test_data_dir);
        shell_set_verbose(true);

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
            load_block->Var("mean");
            std::ofstream fout(main_program_path, std::ios::out | std::ios::binary);
            ASSERT_TRUE(fout);
            fout << main_program->Proto()->SerializeAsString();
            fout.close();
        }
    }

    static void TearDownTestCase()
    {
        std::unique_ptr<FileSystem> fs(CREATE_CLASS(FileSystem, "LocalFileSystem"));
        fs->remove(test_data_dir);
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
    ASSERT_NE(nullptr, executor);
    YAML::Node config = YAML::Load("[1, 2, 3]");
    ASSERT_NE(0, executor->initialize(config, context_ptr));
    config = YAML::Load(string::format_string("{startup_program: %s, main_program: %s}", startup_program_path, main_program_path));
    ASSERT_EQ(0, executor->initialize(config, context_ptr));
    config = YAML::Load(string::format_string("{thread_num: 2, startup_program: %s, main_program: %s}", startup_program_path, main_program_path));
    ASSERT_EQ(0, executor->initialize(config, context_ptr));
}

TEST_F(SimpleExecutorTest, run) {
    std::unique_ptr<Executor> executor(CREATE_CLASS(Executor, "SimpleExecutor"));
    ASSERT_NE(nullptr, executor);

    auto config = YAML::Load(string::format_string("{thread_num: 2, startup_program: %s, main_program: %s}", startup_program_path, main_program_path));
    ASSERT_EQ(0, executor->initialize(config, context_ptr));
    
    auto x_var = executor->mutable_var<::paddle::framework::LoDTensor>("x");
    ASSERT_NE(nullptr, x_var);

    int x_len = 10;
    x_var->Resize({1, x_len});
    auto x_data = x_var->mutable_data<float>(context_ptr->cpu_place);
    ASSERT_NE(nullptr, x_data);
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
