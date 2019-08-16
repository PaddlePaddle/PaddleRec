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
const char feed_path[] = "paddle/fluid/train/custom_trainer/feed";
const char test_data_dir[] = "test_data";
const char main_program_path[] = "test_data/main_program";
const char startup_program_path[] = "test_data/startup_program";
const char model_desc_path[] = "test_data/model.yaml";
}

class CreateProgramsTest : public testing::Test
{
public:
    static void SetUpTestCase()
    {
        std::unique_ptr<FileSystem> fs(CREATE_CLASS(FileSystem, "LocalFileSystem"));
        if (fs->exists("./scripts/create_programs.py")) {
            shell_execute(string::format_string("python ./scripts/create_programs.py ./scripts/example.py %s", test_data_dir));
        } else if (fs->exists(string::format_string("%s/scripts/create_programs.py", feed_path))) {
            shell_execute(string::format_string("python %s/scripts/create_programs.py %s/scripts/example.py %s", feed_path, feed_path, test_data_dir));
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

TEST_F(CreateProgramsTest, example_network) {
    std::unique_ptr<Executor> executor(CREATE_CLASS(Executor, "SimpleExecutor"));
    ASSERT_NE(nullptr, executor);

    auto config = YAML::Load(string::format_string("{thread_num: 2, startup_program: %s, main_program: %s}", startup_program_path, main_program_path));
    auto model_desc = YAML::LoadFile(model_desc_path);
    ASSERT_EQ(0, executor->initialize(config, context_ptr));
    
    std::string input_name = "cvm_input";
    ASSERT_TRUE(model_desc["inputs"]);
    ASSERT_EQ(1, model_desc["inputs"].size());
    ASSERT_TRUE(model_desc["inputs"][0]["name"]);
    ASSERT_TRUE(model_desc["inputs"][0]["shape"]);
    ASSERT_EQ(input_name, model_desc["inputs"][0]["name"].as<std::string>());
    std::vector<int> input_shape = model_desc["inputs"][0]["shape"].as<std::vector<int>>(std::vector<int>());
    ASSERT_EQ(2, input_shape.size());
    ASSERT_EQ(-1, input_shape[0]);
    ASSERT_EQ(4488, input_shape[1]);

    ASSERT_TRUE(model_desc["loss_all"]);
    auto loss_all_name = model_desc["loss_all"].as<std::string>();

    ASSERT_TRUE(model_desc["outputs"]);
    ASSERT_EQ(1, model_desc["outputs"].size());
    ASSERT_TRUE(model_desc["outputs"][0]["name"]);
    ASSERT_TRUE(model_desc["outputs"][0]["shape"]);
    ASSERT_TRUE(model_desc["outputs"][0]["label_name"]);
    ASSERT_TRUE(model_desc["outputs"][0]["loss_name"]);

    auto ctr_output_label_name = model_desc["outputs"][0]["label_name"].as<std::string>();
    auto ctr_output_loss_name = model_desc["outputs"][0]["loss_name"].as<std::string>();
    auto ctr_output_name = model_desc["outputs"][0]["name"].as<std::string>();
    std::vector<int> output_shape = model_desc["outputs"][0]["shape"].as<std::vector<int>>(std::vector<int>());
    ASSERT_EQ(2, output_shape.size());
    ASSERT_EQ(-1, output_shape[0]);
    ASSERT_EQ(1, output_shape[1]);

    auto input_var = executor->mutable_var<::paddle::framework::LoDTensor>(input_name);
    auto label_var = executor->mutable_var<::paddle::framework::LoDTensor>(ctr_output_label_name);
    ASSERT_NE(nullptr, input_var);
    ASSERT_NE(nullptr, label_var);

    input_var->Resize({1, input_shape[1]});
    auto input_data = input_var->mutable_data<float>(context_ptr->cpu_place);
    ASSERT_NE(nullptr, input_data);
    for (int i = 0; i < input_shape[1]; ++i) {
        input_data[i] = 0.1;
    }

    label_var->Resize({1, 1});
    auto label_data = label_var->mutable_data<float>(context_ptr->cpu_place);
    ASSERT_NE(nullptr, label_data);
    label_data[0] = 0.5;

    ASSERT_EQ(0, executor->run());

    auto loss_var = executor->var<::paddle::framework::LoDTensor>(ctr_output_loss_name);
    auto loss = loss_var.data<float>()[0];

    auto loss_all_var = executor->var<::paddle::framework::LoDTensor>(loss_all_name);
    auto loss_all = loss_all_var.data<float>()[0];

    auto ctr_output_var = executor->var<::paddle::framework::LoDTensor>(ctr_output_name);
    auto ctr_output = ctr_output_var.data<float>()[0];

    std::cout << "loss: " << loss << std::endl;
    std::cout << "ctr_output: " << ctr_output << std::endl;
    ASSERT_NEAR(loss, loss_all, 1e-9);
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
