#pragma once
#include <functional>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class Execute {
public:
    Execute() {}
    virtual ~Execute() {}

    //初始化，包括进行训练网络&配置加载工作
    virtual int initialize(YAML::Node& exe_config, 
        std::shared_ptr<TrainerContext> context_ptr) = 0;
    
    //scope 可用于填充&取 var
    virtual ::paddle::framework::Scope* scope() {
        return &_scope;
    }
    //直接取var
    template <class T>
    T* var(const std::string& name) {
        return _scope.Var(name).Get<T>();
    }
    template <class T>
    T* mutable_var(const std::string& name) {
        return _scope.Var(name)->GetMutable<T>();
    }

    //执行n轮训练，每轮回调(epoch_id, _scope)
    virtual int run(uint32_t epoch_num, std::function<void(uint32_t, ::paddle::framework::Scope*)>) = 0;
    
protected:
    ::paddle::framework::Scope _scope;
};
REGISTER_REGISTERER(Execute);

class SimpleExecute : public Execute {
public:
    SimpleExecute() {}
    virtual ~SimpleExecute() {}
    virtual int initialize(YAML::Node& exe_config,
        std::shared_ptr<TrainerContext> context_ptr);
    virtual int run(uint32_t epoch_num, std::function<void(uint32_t, ::paddle::framework::Scope*)>) = 0;
protected:
    ::paddle::framework::Executor _execute; 
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
