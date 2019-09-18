#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"
#include "paddle/fluid/train/custom_trainer/feed/monitor/monitor.h"
#include "paddle/fluid/train/custom_trainer/feed/executor/multi_thread_executor.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

std::once_flag MultiThreadExecutor::_async_delete_flag;
std::shared_ptr<std::thread> MultiThreadExecutor::_async_delete_thread;
paddle::framework::Channel<ScopeExecutorContext*> MultiThreadExecutor::_delete_channel;

int MultiThreadExecutor::initialize(YAML::Node exe_config, 
    std::shared_ptr<TrainerContext> context_ptr) {
    int ret = 0;
    _trainer_context = context_ptr.get();
    _train_data_name = exe_config["train_data_name"].as<std::string>();
    _train_batch_size = exe_config["train_batch_size"].as<int>();

    // 暂未使用，后续各流考虑独立线程池，或设置流数据的优先级
    _input_parse_thread_num = exe_config["input_parse_thread_num"].as<int>();
    _push_gradient_thread_num = exe_config["push_gradient_thread_num"].as<int>();
    _train_thread_num = exe_config["train_thread_num"].as<int>();

    _need_dump_all_model = exe_config["need_dump_all_model"].as<bool>();
    CHECK(_train_thread_num > 0 && _train_batch_size > 0);
    _thread_executors.resize(_train_thread_num);
    auto e_class = exe_config["class"].as<std::string>();
    _train_exe_name = exe_config["name"].as<std::string>();
    if (exe_config["debug_layer_list"]) {
        _debug_layer_list = exe_config["debug_layer_list"].as<std::vector<std::string>>();
    }

    omp_set_num_threads(_train_thread_num);
    #pragma omp parallel for
    for (int i = 0; i < _train_thread_num; ++i) {
        auto* e_ptr = CREATE_INSTANCE(Executor, e_class);
        _thread_executors[i].reset(e_ptr);
        if (e_ptr->initialize(exe_config, context_ptr) != 0) {
            VLOG(0) << "executor initialize failed, name:" << _train_exe_name
                << " class:" << e_class;
            ret = -1;
        }
    }
    CHECK(ret == 0);

    // buffer
    _scope_obj_pool.reset(new paddle::ps::ObjectPool<::paddle::framework::Scope>(
        [this]() -> ::paddle::framework::Scope* {
            auto* scope = new ::paddle::framework::Scope();
            _thread_executors[0]->initialize_scope(scope);
            return scope;
        }, _train_thread_num * 8, 0, _train_thread_num * 8));

    // 模型网络加载
    std::string model_config_path = _trainer_context->file_system->path_join(
        "./model", string::format_string("%s/model.yaml", _train_exe_name.c_str()));
    CHECK(_trainer_context->file_system->exists(model_config_path)) 
        << "miss model config file:" << model_config_path;
    _model_config = YAML::LoadFile(model_config_path);
    _persistables.clear();
    for (const auto& accessor_config : _model_config["input_accessor"]) {
        auto accessor_class = accessor_config["class"].as<std::string>();
        auto* accessor_ptr = CREATE_INSTANCE(DataInputAccessor, accessor_class);
        _input_accessors.emplace_back(accessor_ptr);
        CHECK(accessor_ptr->initialize(accessor_config, context_ptr) == 0)
            << "InputAccessor init Failed, class:" << accessor_class;
        if (accessor_config["table_id"]) {
            auto table_id = accessor_config["table_id"].as<int>();
            if (_table_to_accessors.count(table_id) > 0) {
                _table_to_accessors[table_id].push_back(accessor_ptr);
            } else {
                _table_to_accessors[table_id] = {accessor_ptr};
            }
        }
        CHECK(accessor_ptr->collect_persistables_name(_persistables) == 0)
            << "collect_persistables Failed, class:" << accessor_class;
    }
    std::sort(_persistables.begin(), _persistables.end()); // 持久化变量名一定要排序

    // Monitor组件
    for (const auto& monitor_config : _model_config["monitor"]) {
        auto monitor_class = monitor_config["class"].as<std::string>();
        auto* monitor_ptr = CREATE_INSTANCE(Monitor, monitor_class);
        _monitors.emplace_back(monitor_ptr);
        CHECK(monitor_ptr->initialize(monitor_config, context_ptr) == 0)
            << "Monitor init Failed, class:" << monitor_class;
    }

    // 异步删除池
    std::call_once(_async_delete_flag, [this](){
        _delete_channel = paddle::framework::MakeChannel<ScopeExecutorContext*>();
        _delete_channel->SetBlockSize(32);
        _async_delete_thread.reset(new std::thread([this]{
            std::vector<ScopeExecutorContext*> ctxs;
            while (true) {
                while (_delete_channel->Read(ctxs)) {
                    for (auto* ctx : ctxs) {
                        delete ctx;
                    }
                }
                usleep(200000); // 200ms
            }
        }));
    });
    return ret;
}

int32_t MultiThreadExecutor::save_persistables(const std::string& file_path) {
    auto fs = _trainer_context->file_system;
    auto file_name = fs->path_split(file_path).second;
    fs->remove(file_name);
    auto scope_obj = _scope_obj_pool->get();
    for (size_t i = 0; i < _input_accessors.size(); ++i) {
        _input_accessors[i]->collect_persistables(scope_obj.get());
    }
    framework::ProgramDesc prog;
    auto* block = prog.MutableBlock(0);
    auto* op = block->AppendOp();
    op->SetType("save_combine");
    op->SetInput("X", _persistables);
    op->SetAttr("file_path", file_name);
    op->CheckAttrs();

    platform::CPUPlace place;
    framework::Executor exe(place);
    exe.Run(prog, scope_obj.get(), 0, true, true);
    // exe只能将模型产出在本地，这里通过cp方式兼容其他文件系统
    fs->copy(file_name, file_path);
    return 0;
}

paddle::framework::Channel<DataItem> MultiThreadExecutor::run(
    paddle::framework::Channel<DataItem> input, const DataParser* parser) {

    uint64_t epoch_id = _trainer_context->epoch_accessor->current_epoch_id();
    auto* environment = _trainer_context->environment.get();
    // 输入流
    PipelineOptions input_pipe_option;
    input_pipe_option.need_hold_input_data = true;
    input_pipe_option.batch_size = 1;
    input_pipe_option.input_output_rate = _train_batch_size;
    input_pipe_option.buffer_batch_count = _train_thread_num;
    auto input_pipe = std::make_shared<Pipeline<DataItem, ScopePoolObj>>();
    input_pipe->initialize(input_pipe_option, input, 
        [this, parser](DataItem* item, size_t item_num, 
            ScopePoolObj* scope, size_t* scope_num, size_t thread_idx) -> int {
            *scope_num = 1;
            paddle::platform::Timer timer;
            timer.Start();
            auto scope_obj = _scope_obj_pool->get();   
            auto* scope_context = new ScopeExecutorContext(item_num);
            auto* samples = scope_context->samples();
            for (size_t i = 0; i <item_num; ++i) {
                CHECK(parser->parse_to_sample(item[i], samples[i]) == 0);
            }
            for (size_t i = 0; i < _input_accessors.size(); ++i) {
                _input_accessors[i]->forward(samples, item_num, scope_obj.get());
            }
            timer.Pause();
            scope_context->prepare_cost_ms = timer.ElapsedMS(); 
            int64_t data_for_scope = (int64_t)scope_context;
            ScopeHelper::fill_value(scope_obj.get(), _trainer_context->cpu_place,
                "scope_context", data_for_scope);
            *scope = std::move(scope_obj);
            return 0;
        });
    
    // 训练流
    PipelineOptions train_pipe_option;
    train_pipe_option.input_output_rate = 1;
    train_pipe_option.buffer_batch_count = _train_thread_num;
    auto train_pipe = std::make_shared<Pipeline<ScopePoolObj, ScopePoolObj>>();
    train_pipe->connect_to(*input_pipe, train_pipe_option, 
        [this] (ScopePoolObj* in_items, size_t in_num, 
            ScopePoolObj* out_items, size_t* out_num, size_t thread_idx) -> int {
            auto* executor = _thread_executors[thread_idx].get();
            size_t& out_idx = *out_num;
            for (out_idx = 0; out_idx < in_num; ++out_idx) {
                auto* scope = in_items[out_idx].get();
                auto* scope_ctx = (ScopeExecutorContext*)(*ScopeHelper::get_value<int64_t>(
                    scope, _trainer_context->cpu_place, "scope_context"));
                paddle::platform::Timer timer;
                timer.Start();
                CHECK(executor->run(scope) == 0);
                timer.Pause();
                scope_ctx->executor_cost_ms = timer.ElapsedMS();
                out_items[out_idx] = std::move(in_items[out_idx]);
            }
            return 0;
    });

    // 梯度回传流
    PipelineOptions gradient_pipe_option;
    gradient_pipe_option.input_output_rate = 1;
    gradient_pipe_option.buffer_batch_count = _train_thread_num;
    auto gradient_pipe = std::make_shared<Pipeline<ScopePoolObj, int>>();
    gradient_pipe->connect_to(*train_pipe, gradient_pipe_option, 
        [epoch_id, this] (ScopePoolObj* in_items, size_t in_num, 
            int* out_items, size_t* out_num, size_t thread_idx) -> int {
            size_t& out_idx = *out_num;
            for (out_idx = 0; out_idx < in_num; ++out_idx) {
                paddle::platform::Timer timer;
                timer.Start();
                auto* scope = in_items[out_idx].get();
                auto* scope_ctx = (ScopeExecutorContext*)(*ScopeHelper::get_value<int64_t>(
                    scope, _trainer_context->cpu_place, "scope_context"));
                auto* samples = scope_ctx->samples();
                auto sample_num = scope_ctx->sample_num();
                
                out_items[out_idx] = 0;
                scope_ctx->wait_status.resize(_input_accessors.size());
                for (size_t i = 0; i < _input_accessors.size(); ++i) {
                    scope_ctx->wait_status[i] = _input_accessors[i]->backward(samples, sample_num, scope);
                }
                timer.Pause();
                scope_ctx->push_gradient_cost_ms = timer.ElapsedMS();
                
                // Monitor && Debug
                for (auto& monitor : _monitors) {
                    monitor->add_data(epoch_id, this, scope_ctx);
                }
                if (_debug_layer_list.size() > 0) {
                    for (auto& layer_name : _debug_layer_list) {
                        VLOG(2) << "[Debug][Layer]" << ScopeHelper::to_string(scope, layer_name);
                    }
                }
                 // 所有pipe完成后，再异步回收sample
                _delete_channel->Put(scope_ctx);
            }
            return 0;
    });

    // 等待训练流结束
    std::vector<int> gradient_status;
    while (gradient_pipe->read(gradient_status) > 0) {
    }

    // 输出相关监控&统计项
    for (auto& monitor : _monitors) {
        if (monitor->need_compute_result(epoch_id)) {
            monitor->compute_result();
            ENVLOG_WORKER_MASTER_NOTICE("[Monitor]%s, monitor:%s, result:%s",
                _train_exe_name.c_str(), monitor->get_name().c_str(), monitor->format_result().c_str());
            _trainer_context->monitor_ssm << _train_exe_name << ":" << 
                monitor->get_name() << ":" << monitor->format_result() << ","; 
            monitor->reset();
        }
    }
    return input_pipe->backup_channel();
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
