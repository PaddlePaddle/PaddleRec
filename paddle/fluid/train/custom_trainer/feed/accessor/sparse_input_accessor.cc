#include <math.h>
#include <vector>
#include <utility>
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/common/scope_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/input_data_accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int BaseSparseInputAccessor::initialize(YAML::Node config,
    std::shared_ptr<TrainerContext> context_ptr) {
    CHECK(DataInputAccessor::initialize(config, context_ptr) == 0);
    CHECK(config["input"] && config["input"].Type() == YAML::NodeType::Map);
    for (const auto& input : config["input"]) {
        SparseInputVariable variable;
        variable.name = input.first.as<std::string>();
        variable.gradient_name = paddle::framework::GradVarName(variable.name);
        std::string slots_str = input.second["slots"].as<std::string>();
        std::vector<std::string> slots = paddle::string::split_string(slots_str, ","); 
        variable.slot_idx.resize(UINT16_MAX, -1);
        for (int i = 0; i < slots.size(); ++i) {
            auto slot = (uint16_t)atoi(slots[i].c_str());
            variable.slot_idx[slot] = i;
            variable.slot_list.push_back(slot);
        }
        variable.slot_dim = input.second["slot_dim"].as<int>();
        variable.total_dim = variable.slot_list.size() * variable.slot_dim; 
        _x_variables.push_back(variable);
    }
    return 0;
}

// 取sparse数据
int32_t BaseSparseInputAccessor::forward(SampleInstance* samples,
    size_t num, paddle::framework::Scope* scope) {
    CHECK(num > 0);
    auto* ps_client = _trainer_context->pslib->ps_client();
    auto* value_accessor = ps_client->table_accessor(_table_id);
    size_t key_num = 0;
    for (size_t i = 0; i < num; ++i) {
        key_num += samples[i].features.size();
    }
    std::vector<uint64_t> keys(key_num);
    float** pull_values = new float*[key_num];
    auto pull_value_dim = value_accessor->select_dim();

    // 填入sparseKey Request
    size_t key_idx = 0;
    for (size_t i = 0; i < num; ++i) {
        auto& features = samples[i].features;
        for (auto& feature_item : features) {
            feature_item.weights.resize(pull_value_dim, 0.0);
            keys[key_idx] = feature_item.feature_sign;
            pull_values[key_idx++] = &(feature_item.weights[0]);
        }
    }
    auto pull_status = ps_client->pull_sparse(pull_values, _table_id, keys.data(), key_num);
    auto ret = pull_status.get();
    delete[] pull_values;
    if (ret != 0) {
        return ret;
    }

    auto* runtime_data_ptr = new std::vector<SparseVarRuntimeData>();
    auto& var_runtime_data = *runtime_data_ptr;
    var_runtime_data.resize(_x_variables.size());
    int64_t runtime_data_for_scope = (int64_t)runtime_data_ptr;
    ScopeHelper::fill_value(scope, _trainer_context->cpu_place,
        "sparse_runtime_data", runtime_data_for_scope);
    // Variable空间初始化 
    for (size_t i = 0; i < _x_variables.size(); ++i) {
        const auto& variable = _x_variables[i];
        var_runtime_data[i].row_size = num;
        var_runtime_data[i].total_size = num * variable.total_dim;
        auto* tensor = ScopeHelper::resize_lod_tensor(
            scope, variable.name, {num, variable.total_dim});
        auto* grad_tensor = ScopeHelper::resize_lod_tensor(
            scope, variable.gradient_name, {num, variable.total_dim});
        var_runtime_data[i].variable_data = tensor->mutable_data<float>(_trainer_context->cpu_place);
        var_runtime_data[i].gradient_data = grad_tensor->mutable_data<float>(_trainer_context->cpu_place);
        memset((void*) var_runtime_data[i].variable_data, 0, var_runtime_data[i].total_size * sizeof(float)); 
        memset((void*) var_runtime_data[i].gradient_data, 0, var_runtime_data[i].total_size * sizeof(float)); 
    }
    // 参数填入Variable 
    for (size_t samp_idx = 0; samp_idx < num; ++samp_idx) {
        auto& features = samples[samp_idx].features;
        for (auto& feature_item : features) {
            for (size_t i = 0; i < _x_variables.size(); ++i) {
                auto& variable = _x_variables[i];
                auto slot_idx = variable.slot_idx[feature_item.slot_id]; 
                if (slot_idx < 0) {
                    continue;
                }
                float* item_data =  var_runtime_data[i].variable_data +  
                samp_idx * var_runtime_data[i].total_size + variable.slot_dim * slot_idx; 
                fill_input(item_data, &(feature_item.weights[0]), *value_accessor, variable, samples[samp_idx]);
            }
        }
    }
    // Variable后置处理
    for (size_t i = 0; i < _x_variables.size(); ++i) {
        auto& variable = _x_variables[i];
        post_process_input(var_runtime_data[i].variable_data, variable, samples, num);
    }
    return 0;
}

// 更新spare数据
int32_t BaseSparseInputAccessor::backward(SampleInstance* samples,
    size_t num, paddle::framework::Scope* scope) {
    int64_t runtime_data_for_scope = *ScopeHelper::get_value<int64_t>(
            scope, _trainer_context->cpu_place, "sparse_runtime_data");
    auto* runtime_data_ptr = (std::vector<SparseVarRuntimeData>*)runtime_data_for_scope;
    auto& var_runtime_data = *runtime_data_ptr;
    DoneGuard gurad([runtime_data_ptr](){
        delete runtime_data_ptr;
    });
    if (!_need_gradient) {
        return 0;
    }
    auto* ps_client = _trainer_context->pslib->ps_client();
    auto* value_accessor = ps_client->table_accessor(_table_id);

    size_t key_num = 0;
    for (size_t i = 0; i < num; ++i) {
        key_num += samples[i].features.size();
    }
    std::vector<uint64_t> keys(key_num);
    float** push_values = new float*[key_num];
    auto push_value_dim = value_accessor->update_dim();
        
    size_t key_idx = 0;
    for (size_t samp_idx = 0; samp_idx < num; ++samp_idx) {
        auto& features = samples[samp_idx].features;
        for (auto& feature_item : features) {
            feature_item.gradients.resize(push_value_dim, 0.0);
            for (size_t i = 0; i < _x_variables.size(); ++i) {
                auto& variable = _x_variables[i];
                auto slot_idx = variable.slot_idx[feature_item.slot_id]; 
                if (slot_idx < 0) {
                    continue;
                }
                const float* grad_data = var_runtime_data[i].gradient_data +  
                    samp_idx * var_runtime_data[i].total_size + variable.slot_dim * slot_idx; 
                fill_gradient(&(feature_item.gradients[0]), grad_data, 
                    *value_accessor, variable, samples[samp_idx]);
                keys[key_idx] = feature_item.feature_sign;
                push_values[key_idx++] = &(feature_item.gradients[0]);
            }
        }
    }
    auto push_status = ps_client->push_sparse(_table_id, 
        keys.data(), (const float**)push_values, key_num);
    auto ret = push_status.get();
    delete[] push_values;
    return ret;
} 

class AbacusSparseJoinAccessor : public BaseSparseInputAccessor {
public:
    AbacusSparseJoinAccessor() {}
    virtual ~AbacusSparseJoinAccessor() {}
    virtual void fill_input(float* var_data, const float* pull_raw,
        paddle::ps::ValueAccessor& value_accessor, 
        SparseInputVariable& variable, SampleInstance& sample) {
        for (size_t i = 0; i < variable.slot_dim; ++i) {
            var_data[i] += pull_raw[i];
        }
    }

    virtual void post_process_input(float* var_data, 
        SparseInputVariable& variable, SampleInstance* samples, size_t num) {
        for (size_t i = 0; i < num * variable.slot_list.size(); ++i) {
            var_data[0] = log(var_data[0] + 1);                  // show
            var_data[1] = log(var_data[1] + 1) - var_data[0];    // ctr
            var_data += variable.slot_dim; 
        }
    }

    virtual void fill_gradient(float* push_value, const float* gradient_raw,
        paddle::ps::ValueAccessor& value_accessor, 
        SparseInputVariable& variable, SampleInstance& sample) {
        // join阶段不回填梯度
        CHECK(false);
        return;
    }
};
REGIST_CLASS(DataInputAccessor, AbacusSparseJoinAccessor);

class AbacusSparseUpdateAccessor : public BaseSparseInputAccessor {
public:
    AbacusSparseUpdateAccessor() {}
    virtual ~AbacusSparseUpdateAccessor() {}
    virtual void fill_input(float* var_data, const float* pull_raw,
        paddle::ps::ValueAccessor& value_accessor, 
        SparseInputVariable& variable, SampleInstance& sample) {
        for (size_t i = 0; i < variable.slot_dim; ++i) {
            var_data[i] += pull_raw[i + 2];
        }
    }

    virtual void post_process_input(float* var_data, 
        SparseInputVariable& variable, SampleInstance* samples, size_t num) {
        return;
    }

    virtual void fill_gradient(float* push_value, const float* gradient_raw,
        paddle::ps::ValueAccessor& value_accessor, 
        SparseInputVariable& variable, SampleInstance& sample) {
        push_value[0] += 1;
        push_value[1] += sample.labels[0];
        for (size_t i = 0; i < variable.slot_dim; ++i) {
            push_value[i + 2] += gradient_raw[i];
        }
        return;
    }
};
REGIST_CLASS(DataInputAccessor, AbacusSparseUpdateAccessor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
