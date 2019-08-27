#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>

#include "paddle/fluid/train/custom_trainer/feed/executor/executor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"
#include "paddle/fluid/train/custom_trainer/feed/io/shell.h"
#include "paddle/fluid/train/custom_trainer/feed/common/scope_helper.h"
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
        std::unique_ptr<FileSystem> fs(CREATE_INSTANCE(FileSystem, "LocalFileSystem"));
        if (fs->exists("./scripts/create_programs.py")) {
            shell_execute(string::format_string("python ./scripts/create_programs.py ./scripts/example.py %s", test_data_dir));
        } else if (fs->exists(string::format_string("%s/scripts/create_programs.py", feed_path))) {
            shell_execute(string::format_string("python %s/scripts/create_programs.py %s/scripts/example.py %s", feed_path, feed_path, test_data_dir));
        }
    }

    static void TearDownTestCase()
    {
        std::unique_ptr<FileSystem> fs(CREATE_INSTANCE(FileSystem, "LocalFileSystem"));
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

    float random(float min_x = 0.0, float max_x = 1.0) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        return min_x + (max_x - min_x) * r;
    }

    std::shared_ptr<TrainerContext> context_ptr;
};

TEST_F(CreateProgramsTest, example_network) {
    std::unique_ptr<Executor> executor(CREATE_INSTANCE(Executor, "SimpleExecutor"));
    ASSERT_NE(nullptr, executor);

    auto config = YAML::Load(string::format_string("{thread_num: 2, startup_program: %s, main_program: %s}", startup_program_path, main_program_path));
    auto model_desc = YAML::LoadFile(model_desc_path);
    ASSERT_EQ(0, executor->initialize(config, context_ptr));

    std::string input_name = "cvm_input";
    std::string loss_name = "loss_ctr";
    std::string label_name = "label_ctr";

    // loss
    ASSERT_TRUE(model_desc["loss"]);
    ASSERT_EQ(loss_name, model_desc["loss"].as<std::string>());

    // input
    ASSERT_TRUE(model_desc["inputs"]);
    ASSERT_EQ(1, model_desc["inputs"].size());
    ASSERT_TRUE(model_desc["inputs"][0]["name"]);
    ASSERT_TRUE(model_desc["inputs"][0]["shape"]);
    ASSERT_EQ(input_name, model_desc["inputs"][0]["name"].as<std::string>());
    auto input_shape = model_desc["inputs"][0]["shape"].as<std::vector<int>>(std::vector<int>());
    ASSERT_EQ(2, input_shape.size());
    ASSERT_EQ(-1, input_shape[0]);
    ASSERT_EQ(4488, input_shape[1]);

    // label
    ASSERT_TRUE(model_desc["labels"]);
    ASSERT_EQ(1, model_desc["labels"].size());
    ASSERT_TRUE(model_desc["labels"][0]["name"]);
    ASSERT_TRUE(model_desc["labels"][0]["shape"]);
    ASSERT_EQ(label_name, model_desc["labels"][0]["name"].as<std::string>());
    auto label_shape = model_desc["labels"][0]["shape"].as<std::vector<int>>(std::vector<int>());
    ASSERT_EQ(2, label_shape.size());
    ASSERT_EQ(-1, label_shape[0]);
    ASSERT_EQ(1, label_shape[1]);

    ASSERT_TRUE(model_desc["outputs"]);
    ASSERT_EQ(1, model_desc["outputs"].size());
    ASSERT_TRUE(model_desc["outputs"][0]["name"]);
    ASSERT_TRUE(model_desc["outputs"][0]["shape"]);
    auto output_name = model_desc["outputs"][0]["name"].as<std::string>();
    auto output_shape = model_desc["outputs"][0]["shape"].as<std::vector<int>>(std::vector<int>());
    ASSERT_EQ(2, output_shape.size());
    ASSERT_EQ(-1, output_shape[0]);
    ASSERT_EQ(1, output_shape[1]);

    paddle::framework::Scope scope;
    executor->initialize_scope(&scope);
    auto input_var = ScopeHelper::mutable_var<::paddle::framework::LoDTensor>(&scope, input_name);
    auto label_var = ScopeHelper::mutable_var<::paddle::framework::LoDTensor>(&scope, label_name);
    ASSERT_NE(nullptr, input_var);
    ASSERT_NE(nullptr, label_var);

    input_var->Resize({1, input_shape[1]});
    auto input_data = input_var->mutable_data<float>(context_ptr->cpu_place);
    ASSERT_NE(nullptr, input_data);
    for (int i = 0; i < input_shape[1]; ++i) {
        input_data[i] = random();
    }

    label_var->Resize({1, 1});
    auto label_data = label_var->mutable_data<float>(context_ptr->cpu_place);
    ASSERT_NE(nullptr, label_data);
    label_data[0] = random();

    ASSERT_EQ(0, executor->run(&scope));

    auto loss_var = ScopeHelper::var<::paddle::framework::LoDTensor>(&scope, loss_name);
    auto loss = loss_var.data<float>()[0];

    auto output_var = ScopeHelper::var<::paddle::framework::LoDTensor>(&scope, output_name);
    auto output = output_var.data<float>()[0];

    LOG(INFO) << "loss: " << loss << std::endl;
    LOG(INFO) << "label: " << label_data[0] << std::endl;
    LOG(INFO) << "output: " << output << std::endl;
    ASSERT_NEAR(loss, pow(output - label_data[0], 2), 1e-8);
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
