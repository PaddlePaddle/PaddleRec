#include "paddle/fluid/train/custom_trainer/feed/accessor/input_data_accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class WeightsAdjustAccessor : public DataInputAccessor {
public:
    WeightsAdjustAccessor() {}
    virtual ~WeightsAdjustAccessor() {}
    
    virtual int initialize(YAML::Node config,
         std::shared_ptr<TrainerContext> context_ptr) {
        _trainer_context = context_ptr.get();
        _slot_id = config["slot_id"].as<int>();
        _input_name = config["input"].as<std::string>();
        _adjw_ratio = config["adjw_ratio"].as<float>();
        _adjw_threshold = config["adjw_threshold"].as<float>();
        return 0;
    }

    virtual int32_t forward(SampleInstance* samples, size_t num,
        ::paddle::framework::Scope* scope) {
        int64_t runtime_data_for_scope = *ScopeHelper::get_value<int64_t>(
            scope, _trainer_context->cpu_place, "sparse_runtime_data");
        auto* runtime_data_ptr = (std::vector<SparseVarRuntimeData>*)runtime_data_for_scope;
        auto& var_runtime_data = *runtime_data_ptr;
        
        int slot_idx = -1;
        SparseVarRuntimeData* sparse_var_data = nullptr;
        for (auto& sparse_var : var_runtime_data) {
            slot_idx = sparse_var.sparse_var_metas->slot_idx[_slot_id];
            if (slot_idx >= 0) {
                sparse_var_data = &sparse_var;
                break;
            }
        }
        CHECK(slot_idx >= 0) << "Not Found this Slot in slot_list. slot_id:" << _slot_id;
        
        auto* tensor = ScopeHelper::resize_lod_tensor(scope, _input_name, {num, 1});
         auto* weights_data = tensor->mutable_data<float>(_trainer_context->cpu_place);
        
        float* sparse_input_data = sparse_var_data->variable_data;
        size_t sparse_slot_dim = sparse_var_data->sparse_var_metas->slot_dim;
        size_t sparse_input_col = sparse_var_data->sparse_var_metas->total_dim;
        for (int i = 0; i < num; ++i) {
            float show = sparse_input_data[i * sparse_input_col + slot_idx * sparse_slot_dim];
            show = pow(M_E, show) - 1; // show在fill时算过log，这里恢复原值
            weights_data[i] = 1.0;
            if (show >= 0 && show < _adjw_threshold) {
                weights_data[i] = log(M_E + (_adjw_threshold - show) / _adjw_threshold * _adjw_ratio);
            }
        } 
         return 0;
    }

    virtual std::future<int32_t> backward(SampleInstance* samples, size_t num,
        ::paddle::framework::Scope* scope) {
        std::future<int32_t> ret;
        return ret;
    }
protected:
    size_t _slot_id;
    float _adjw_ratio;
    float _adjw_threshold; 
    std::string _input_name;
};

REGIST_CLASS(DataInputAccessor, WeightsAdjustAccessor);


}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
