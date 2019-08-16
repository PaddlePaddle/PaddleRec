#pragma once
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class DataInputAccessor : public Accessor {
public:
    DataInputAccessor() {}
    virtual ~DataInputAccessor() {}

    virtual int initialize(const YAML::Node& config,
        std::shared_ptr<TrainerContext> context_ptr);

    // 前向， 一般用于填充输入，在训练网络执行前调用
    virtual int32_t forward(const SampleInstance* samples,
        ::paddle::framework::Scope* scope, size_t table_id, size_t num) = 0;

    // 后向，一般用于更新梯度，在训练网络执行后调用
    virtual int32_t backward(const SampleInstance* samples,
        ::paddle::framework::Scope* scope, size_t table_id, size_t num) = 0;
protected:
};
REGIST_REGISTERER(DataInputAccessor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
