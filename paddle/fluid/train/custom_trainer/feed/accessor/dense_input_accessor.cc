#include <sstream>
#include "gflags/gflags.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/input_data_accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

DEFINE_string(feed_trainer_debug_dense_name, "", "open dense debug for specif layer_name");
    
int DenseInputAccessor::initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
    CHECK(DataInputAccessor::initialize(config, context_ptr) == 0);
    _total_dim = 0;
    _pull_request_num.store(0);
    for (const auto& input : config["input"]) {
        DenseInputVariable variable;
        variable.name = input["name"].as<std::string>();
        variable.gradient_name = paddle::framework::GradVarName(variable.name);
        variable.shape = input["shape"].as<std::vector<int>>();
        variable.dim = 1;
        for (int i = 0; i < variable.shape.size(); ++i) {
            if (variable.shape[i] <= 0) {
                variable.shape[i] = 1;
            }
            variable.dim *= variable.shape[i];    
        }
        _total_dim += variable.dim;
        _x_variables.emplace_back(variable);
    }
    if (config["async_pull"] && config["async_pull"].as<bool>()) {
        _need_async_pull = true;
    }
    return 0;
}

int32_t DenseInputAccessor::create(::paddle::framework::Scope* scope) {
    size_t data_buffer_idx = 0;
    std::vector<paddle::ps::Region> regions;
    for (auto& variable : _x_variables) {
        auto* tensor = scope->Var(variable.name)->
            GetMutable<paddle::framework::LoDTensor>(); 
        auto* data = tensor->data<float>();
        regions.emplace_back(data, variable.dim);
        if (FLAGS_feed_trainer_debug_dense_name == variable.name) 
            VLOG(2) << "[Debug][CreateDense]" <<  ScopeHelper::to_string(scope, variable.name); 
    }
    auto* ps_client = _trainer_context->pslib->ps_client();
    auto push_status = ps_client->push_dense_param(regions.data(), regions.size(), _table_id);
    return push_status.get();
}

// rpc拉取数据，需保证单线程运行
int32_t DenseInputAccessor::pull_dense(size_t table_id) {
    float* data_buffer =  _data_buffer;
    if (data_buffer == NULL) {
        data_buffer = new float[_total_dim];
    }
    size_t data_buffer_idx = 0;
    std::vector<paddle::ps::Region> regions;
    for (auto& variable : _x_variables) {
        regions.emplace_back(data_buffer + data_buffer_idx, variable.dim);
        data_buffer_idx += variable.dim;
    }
    auto* ps_client = _trainer_context->pslib->ps_client();
    auto push_status = ps_client->pull_dense(regions.data(), regions.size(), table_id);
    int32_t ret = push_status.get();
    // TODO 使用双buffer DataBuffer,避免训练期改写，当前异步SGD下，问题不大
    _data_buffer = data_buffer;
    return ret;
}

int32_t DenseInputAccessor::forward(SampleInstance* samples, size_t num,
    paddle::framework::Scope* scope) {
    collect_persistables(scope);

    if (_need_async_pull) {
        ++_pull_request_num;
    }
    return 0;
}

int32_t DenseInputAccessor::collect_persistables(paddle::framework::Scope* scope) {
    // 首次同步pull，之后异步pull
    if (_data_buffer == nullptr) {
        _pull_mutex.lock();
        if (_data_buffer == nullptr) {
            CHECK(pull_dense(_table_id) == 0);
            _async_pull_thread = std::make_shared<std::thread>(
                [this]() {
                while (_need_async_pull) {
                    if (_pull_request_num > 0) {
                        pull_dense(_table_id);
                        _pull_request_num = 0; 
                    } else {
                        usleep(50000);
                    }     
                }
            });
        }
        _pull_mutex.unlock();
    }
    size_t data_buffer_idx = 0;
    for (auto& variable : _x_variables) {
        auto* shape_ptr = &(variable.shape[0]);
        paddle::framework::DDim ddim(shape_ptr, variable.shape.size());
        auto* tensor = ScopeHelper::resize_lod_tensor(scope, variable.name, ddim);  
        auto* grad_tensor = ScopeHelper::resize_lod_tensor(scope, variable.gradient_name, ddim);
        VLOG(5) << "fill scope variable:" << variable.name << ", " << variable.gradient_name
            << ", data_buffer: " << _data_buffer + data_buffer_idx
            << ", dim: " << variable.dim * sizeof(float);
        auto* var_data = tensor->mutable_data<float>(_trainer_context->cpu_place);
        memcpy(var_data, _data_buffer + data_buffer_idx, variable.dim * sizeof(float));
        data_buffer_idx += variable.dim;
    }
    if (!FLAGS_feed_trainer_debug_dense_name.empty()) {
        for (auto& variable : _x_variables) {
            if (variable.name != FLAGS_feed_trainer_debug_dense_name) {
                continue;
            }
            VLOG(2) << "[Debug][PullDense]" << ScopeHelper::to_string(scope, variable.name);
        }
    }
    return 0;
}

int32_t DenseInputAccessor::collect_persistables_name(std::vector<std::string>& persistables) {
    for (auto& variable : _x_variables) {
        persistables.push_back(variable.name);
    }
    return 0;
}

std::future<int32_t> DenseInputAccessor::backward(SampleInstance* samples, size_t num,
        paddle::framework::Scope* scope) {
    std::future<int32_t> ret;
    if (!_need_gradient) {
        return ret;
    }
    size_t data_buffer_idx = 0;
    std::vector<paddle::ps::Region> regions;
    for (auto& variable : _x_variables) {
        auto* tensor = scope->Var(variable.gradient_name)->
            GetMutable<paddle::framework::LoDTensor>();
        auto* grad_data = tensor->mutable_data<float>(_trainer_context->cpu_place);
        regions.emplace_back(grad_data, variable.dim);
    }
    auto* ps_client = _trainer_context->pslib->ps_client();
    ps_client->push_dense(regions.data(), regions.size(), _table_id);
    if (!FLAGS_feed_trainer_debug_dense_name.empty()) {
        for (auto& variable : _x_variables) {
            if (variable.name != FLAGS_feed_trainer_debug_dense_name) {
                continue;
            }
            VLOG(2) << "[Debug][PushDense]" << ScopeHelper::to_string(scope, variable.gradient_name);
        }
    }
    // not wait dense push
    return ret;
}

int32_t EbdVariableInputAccessor::forward(SampleInstance* samples, size_t num,
    paddle::framework::Scope* scope) {
    CHECK(_x_variables.size() == 1);
    CHECK(_x_variables[0].shape.size() == 1);
    auto& variable = _x_variables[0];
    auto* tensor = ScopeHelper::resize_lod_tensor(scope, 
        variable.name, {num, variable.shape[0]});
    auto* var_data = tensor->mutable_data<float>(_trainer_context->cpu_place);
    for (size_t i = 0; i < num; ++i) {
        auto& sample = samples[i];
        CHECK(sample.embedx.size() == variable.dim);
        memcpy(var_data, sample.embedx.data(), variable.dim * sizeof(float));
        var_data += variable.dim;
    }
    return 0;
}
std::future<int32_t> EbdVariableInputAccessor::backward(SampleInstance* samples, size_t num,
    paddle::framework::Scope* scope) {
    std::future<int32_t> ret;
    return ret;
}

REGIST_CLASS(DataInputAccessor, DenseInputAccessor);
REGIST_CLASS(DataInputAccessor, EbdVariableInputAccessor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
