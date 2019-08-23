#include "paddle/fluid/train/custom_trainer/feed/accessor/input_data_accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {
    
int DenseInputAccessor::initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
    CHECK(DataInputAccessor::initialize(config, context_ptr) == 0);
    CHECK(config["input"] && config["input"].Type() == YAML::NodeType::Map);
    _total_dim = 0;
    _pull_request_num.store(0);
    for (const auto& input : config["input"]) {
        DenseInputVariable variable;
        variable.name = input.first.as<std::string>();
        variable.gradient_name = paddle::framework::GradVarName(variable.name);
        variable.shape = input.second["shape"].as<std::vector<int>>();
        variable.dim = 1;
        for (int i = 0; i < variable.shape.size(); ++i) {
            if (variable.shape[i] <= 0) {
                variable.shape[i] = 1;
            }
            variable.dim *= variable.shape[i];    
        }
        _total_dim += variable.dim;
    }
    if (config["async_pull"] && config["async_pull"].as<bool>()) {
        _need_async_pull = true;
    }
    return 0;
}

// rpc拉取数据，需保证单线程运行
int32_t DenseInputAccessor::pull_dense(size_t table_id) {
    float* data_buffer = NULL;
    if (_data_buffer != nullptr) {
        data_buffer = _data_buffer;
    } else {
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
    return push_status.get();
}

int32_t DenseInputAccessor::forward(SampleInstance* samples, size_t num,
    paddle::framework::Scope* scope) {
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
        auto* var_data = tensor->mutable_data<float>(_trainer_context->cpu_place);
        memcpy(var_data, _data_buffer + data_buffer_idx, variable.dim * sizeof(float));
        data_buffer_idx += variable.dim;
    }
    if (_need_async_pull) {
        ++_pull_request_num;
    }
    return 0;
}

int32_t DenseInputAccessor::backward(SampleInstance* samples, size_t num,
        paddle::framework::Scope* scope) {
    if (!_need_gradient) {
        return 0;
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
    auto push_status = ps_client->push_dense(regions.data(), regions.size(), _table_id);
    return push_status.get();
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

int32_t EbdVariableInputAccessor::backward(SampleInstance* samples, size_t num,
    paddle::framework::Scope* scope) {
    return 0;
}

REGIST_CLASS(DataInputAccessor, DenseInputAccessor);
REGIST_CLASS(DataInputAccessor, EbdVariableInputAccessor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
