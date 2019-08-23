#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"
#include "paddle/fluid/train/custom_trainer/feed/executor/multi_thread_executor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int MultiThreadExecutor::initialize(YAML::Node exe_config, 
    std::shared_ptr<TrainerContext> context_ptr) {
    int ret = 0;
    _trainer_context = context_ptr.get();
    _train_data_name = exe_config["train_data_name"].as<std::string>();
    _train_batch_size = exe_config["train_batch_size"].as<int>();
    _input_parse_thread_num = exe_config["input_parse_thread_num"].as<int>();
    _push_gradient_thread_num = exe_config["push_gradient_thread_num "].as<int>();
    _train_thread_num = exe_config["train_thread_num"].as<int>();
    _need_dump_all_model = exe_config["need_dump_all_model"].as<bool>();
    CHECK(_train_thread_num > 0 && _train_batch_size > 0);
    _thread_executors.resize(_train_thread_num);
    auto e_class = exe_config["class"].as<std::string>();
    _train_exe_name = exe_config["name"].as<std::string>();

    omp_set_num_threads(_train_thread_num);
    #pragma omp parallel for
    for (int i = 0; i < _train_thread_num; ++i) {
        auto* e_ptr = CREATE_INSTANCE(Executor, e_class);
        _thread_executors[i].reset(e_ptr);
        if (e_ptr->initialize(exe_config, context_ptr) != 0) {
            ret = -1;
        }
    }
    CHECK(ret == 0);
    _scope_obj_pool.config(
        [this]() -> ::paddle::framework::Scope* {
            auto* scope = new ::paddle::framework::Scope();
            _thread_executors[0]->initialize_scope(scope);
            return scope;
        }, _train_thread_num * 8);

    std::string model_config_path = _trainer_context->file_system->path_join(
        "./model", string::format_string("%s.yaml", _train_exe_name.c_str()));
    CHECK(_trainer_context->file_system->exists(model_config_path)) 
        << "miss model config file:" << model_config_path;
    _model_config = YAML::LoadFile(model_config_path);
    _input_accessors.resize(_model_config["input_accessor"].size());
    for (const auto& accessor_config : _model_config["input_accessor"]) {
        auto accessor_class = accessor_config["class"].as<std::string>();
        _input_accessors.emplace_back(CREATE_INSTANCE(DataInputAccessor, accessor_class));
        CHECK(_input_accessors.back()->initialize(accessor_config, context_ptr) == 0)
            << "InputAccessor init Failed, class:" << accessor_class;
    } 

    return ret;
}

paddle::framework::Channel<DataItem> MultiThreadExecutor::run(
    paddle::framework::Channel<DataItem> input, const DataParser* parser) {
    PipelineOptions input_pipe_option;
    input_pipe_option.need_hold_input_data = true;
    input_pipe_option.batch_size = _train_batch_size;
    input_pipe_option.thread_num = _input_parse_thread_num;
    input_pipe_option.input_output_rate = _train_batch_size;
    input_pipe_option.buffer_batch_count = _train_thread_num;
    auto input_pipe = std::make_shared<Pipeline<DataItem, ScopePoolObj>>();
    input_pipe->initialize(input_pipe_option, input, 
        [this, parser](DataItem* item, size_t item_num, 
            ScopePoolObj* scope, size_t* scope_num, size_t thread_idx) -> int {
            *scope_num = 1;
            auto scope_obj = _scope_obj_pool.get();   
            auto* samples = new SampleInstance[item_num];
            for (size_t i = 0; i <item_num; ++i) {
                CHECK(parser->parse_to_sample(item[i], samples[i]) == 0);
            }
            for (size_t i = 0; i < _input_accessors.size(); ++i) {
                _input_accessors[i]->forward(samples, item_num, scope_obj.get());
            }
            int64_t data_for_scope = (int64_t)samples;
            ScopeHelper::fill_value(scope_obj.get(), _trainer_context->cpu_place,
                "sample_data", data_for_scope);
            data_for_scope = (int64_t)item_num;
            ScopeHelper::fill_value(scope_obj.get(), _trainer_context->cpu_place,
                "sample_num", data_for_scope);
            *scope = std::move(scope_obj);
            return 0;
        });
    
    PipelineOptions train_pipe_option;
    train_pipe_option.input_output_rate = 1;
    train_pipe_option.thread_num = _train_thread_num;
    train_pipe_option.buffer_batch_count = 2 * _train_thread_num;
    auto train_pipe = std::make_shared<Pipeline<ScopePoolObj, ScopePoolObj>>();
    train_pipe->connect_to(*input_pipe, train_pipe_option, 
        [this] (ScopePoolObj* in_items, size_t in_num, 
            ScopePoolObj* out_items, size_t* out_num, size_t thread_idx) -> int {
            auto* executor = _thread_executors[thread_idx].get();
            size_t& out_idx = *out_num;
            for (out_idx = 0; out_idx < in_num; ++out_idx) {
                executor->run(in_items[out_idx].get());
                out_items[out_idx] = std::move(in_items[out_idx]);
            }
            return 0;
        });

    PipelineOptions gradient_pipe_option;
    gradient_pipe_option.input_output_rate = 1;
    gradient_pipe_option.thread_num = _push_gradient_thread_num;
    gradient_pipe_option.buffer_batch_count = 2 * _train_thread_num;
    auto gradient_pipe = std::make_shared<Pipeline<ScopePoolObj, int>>();
    gradient_pipe->connect_to(*train_pipe, gradient_pipe_option, 
        [this] (ScopePoolObj* in_items, size_t in_num, 
            int* out_items, size_t* out_num, size_t thread_idx) -> int {
            size_t& out_idx = *out_num;
            for (out_idx = 0; out_idx < in_num; ++out_idx) {
                auto* scope = in_items[out_idx].get();
                auto sample_num = *ScopeHelper::get_value<int64_t>(
                    scope, _trainer_context->cpu_place, "sample_num");
                
                auto* samples = (SampleInstance*)(*ScopeHelper::get_value<int64_t>(
                    scope, _trainer_context->cpu_place, "sample_data"));
                for (size_t i = 0; i < _input_accessors.size(); ++i) {
                    out_items[out_idx] = _input_accessors[i]->
                        backward(samples, sample_num, scope);
                }
                delete[] samples; // 所有pipe完成后，再回收sample
            }

            return 0;
        });

    std::vector<int> gradient_status;
    while (gradient_pipe->read(gradient_status) > 0) {
    }
    return input_pipe->backup_channel();
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
