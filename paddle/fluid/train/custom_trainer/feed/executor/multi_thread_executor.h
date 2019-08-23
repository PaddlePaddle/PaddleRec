#pragma once
#include <functional>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/train/custom_trainer/feed/executor/executor.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/input_data_accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class MultiThreadExecutor {
public:
    MultiThreadExecutor() {}
    virtual ~MultiThreadExecutor() {}

    //初始化，包括进行训练网络&配置加载工作
    virtual int initialize(YAML::Node exe_config, 
        std::shared_ptr<TrainerContext> context_ptr);

    //执行训练
    virtual paddle::framework::Channel<DataItem> run(
        paddle::framework::Channel<DataItem> input, const DataParser* parser);
    
    virtual bool is_dump_all_model() {
        return _need_dump_all_model;
    }

    virtual const std::string& train_data_name() {
        return _train_data_name;
    }
protected:
    std::string _train_data_name;
    size_t _train_batch_size = 32;
    size_t _train_thread_num = 12;
    size_t _input_parse_thread_num = 10;
    size_t _push_gradient_thread_num = 10;
    bool _need_dump_all_model = false;

    YAML::Node _model_config;
    std::string _train_exe_name;
    TrainerContext* _trainer_context = nullptr;
    std::vector<std::shared_ptr<Executor>> _thread_executors;
    std::vector<std::shared_ptr<DataInputAccessor>> _input_accessors;
    paddle::ps::ObjectPool<::paddle::framework::Scope> _scope_obj_pool;
    typedef paddle::ps::ObjectPool<::paddle::framework::Scope>::PooledObject ScopePoolObj;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
