#pragma once
#include <functional>
#include "paddle/fluid/framework/scope.h"
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
    const T& var(const std::string& name) {
        return _scope.Var(name)->Get<T>();
    }
    template <class T>
    T* mutable_var(const std::string& name) {
        return _scope.Var(name)->GetMutable<T>();
    }

    //执行训练
    virtual int run() = 0;
    
protected:
    ::paddle::framework::Scope _scope;
};
REGISTER_REGISTERER(Execute);

class SimpleExecute : public Execute {
public:
    SimpleExecute();
    virtual ~SimpleExecute();
    virtual int initialize(YAML::Node& exe_config,
        std::shared_ptr<TrainerContext> context_ptr);
    virtual int run();
protected:
    struct Context;
    std::unique_ptr<Context> _context;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
