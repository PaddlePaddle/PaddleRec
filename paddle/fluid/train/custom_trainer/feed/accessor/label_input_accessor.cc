#include "paddle/fluid/train/custom_trainer/feed/accessor/input_data_accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {
    
int LabelInputAccessor::initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
    _trainer_context = context_ptr.get();
    _label_total_dim = 0;
    for (const auto& input : config["input"]) {
        LabelInputVariable variable;
        variable.label_name = input["label_name"].as<std::string>();
        variable.output_name = input["output_name"].as<std::string>();
        auto shape = input["shape"].as<std::vector<int>>();
        variable.label_dim = 0;
        for (auto dim : shape) {
            variable.label_dim += (dim > 0 ? dim : 0);
        }
        _label_total_dim += variable.label_dim;
        _labels.emplace_back(variable);
    }
    return 0;
}

int32_t LabelInputAccessor::forward(SampleInstance* samples, size_t num,
    paddle::framework::Scope* scope) {
    if (num < 1) {
        return 0;
    }
    size_t sample_label_data_idx = 0;
    for (auto& label : _labels) {
        auto* tensor = ScopeHelper::resize_lod_tensor(scope, label.label_name, {num, label.label_dim}); 
        auto* res_tens = ScopeHelper::resize_lod_tensor(scope, label.output_name, {num, label.label_dim}); 
        auto* var_data = tensor->mutable_data<float>(_trainer_context->cpu_place);        
        for (size_t i = 0; i < num; ++i) {
            auto& sample = samples[i];
            CHECK(sample.labels.size() > sample_label_data_idx);
            float* sample_label_buffer = sample.labels.data();
            memcpy(var_data + i * label.label_dim, 
                sample_label_buffer + sample_label_data_idx, label.label_dim * sizeof(float));
        }
        sample_label_data_idx += label.label_dim;  
    }
    return 0;
}

int32_t LabelInputAccessor::backward(SampleInstance* samples, size_t num,
        paddle::framework::Scope* scope) {
    if (num < 1) {
        return 0;
    }
    for (size_t i = 0; i < num; ++i) {
        auto& sample = samples[i];
        sample.predicts.resize(_label_total_dim);
        size_t sample_predict_data_idx = 0;
        float* sample_predict_buffer = sample.predicts.data();
        for (auto& label : _labels) {
            auto* tensor = scope->Var(label.output_name)->
                GetMutable<paddle::framework::LoDTensor>(); 
            auto* var_data = tensor->mutable_data<float>(_trainer_context->cpu_place);        
            memcpy(sample_predict_buffer + sample_predict_data_idx, 
                var_data + i * label.label_dim, label.label_dim * sizeof(float));
            sample_predict_data_idx += label.label_dim;  
        }
    }
    return 0;
}

REGIST_CLASS(DataInputAccessor, LabelInputAccessor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
