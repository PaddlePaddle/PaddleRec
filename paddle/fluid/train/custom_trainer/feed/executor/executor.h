#pragma once
#include <functional>
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class Executor {
public:
    Executor() {}
    virtual ~Executor() {}

    // 初始化，包括进行训练网络&配置加载工作
    virtual int initialize(YAML::Node exe_config, 
        std::shared_ptr<TrainerContext> context_ptr) = 0;
    
    // 初始化scope, 后续反复执行训练，不再初始化
    virtual int initialize_scope(::paddle::framework::Scope* scope) = 0;

    // 执行训练
    virtual int run(::paddle::framework::Scope* scope) = 0;

    // cost time millisecond
    virtual uint64_t epoch_cost() const {
        return 0;
    }
protected:
    ::paddle::framework::Scope _scope;
};
REGIST_REGISTERER(Executor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
