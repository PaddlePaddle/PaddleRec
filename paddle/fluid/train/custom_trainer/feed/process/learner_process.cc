/*
 *Author: xiexionghang
 *Train样本
 */
#include <omp.h>
#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"
#include "paddle/fluid/train/custom_trainer/feed/dataset/dataset.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/epoch_accessor.h"
#include "paddle/fluid/train/custom_trainer/feed/process/learner_process.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int LearnerProcess::initialize(std::shared_ptr<TrainerContext> context_ptr) {
    int ret = Process::initialize(context_ptr);
    auto& config = _context_ptr->trainer_config;
    if (config["executor"]) {
        _executors.resize(config["executor"].size());
        for (size_t i = 0; i < _executors.size(); ++i) {
            _executors[i].reset(new MultiThreadExecutor());
            CHECK(_executors[i]->initialize(config["executor"][i], context_ptr) == 0);
        }
    }
    return 0;
}

std::future<int> LearnerProcess::save_model(uint64_t epoch_id, int table_id, ModelSaveWay way) {
    std::promise<int> p;
    auto ret = p.get_future();
    auto* ps_client = _context_ptr->pslib->ps_client();
    auto* epoch_accessor = _context_ptr->epoch_accessor.get();
    if (epoch_accessor->need_save_model(epoch_id, way)) {
        VLOG(2) << "Start save model, table_id:" << table_id;
        auto model_dir = epoch_accessor->model_save_path(epoch_id, way);
        return ps_client->save(table_id, model_dir, std::to_string((int)way));
    } else {
        p.set_value(0);
    }
    return ret;
}

int LearnerProcess::wait_save_model(uint64_t epoch_id, ModelSaveWay way) {
    auto* environment = _context_ptr->environment.get();
    if (!environment->is_master_node(EnvironmentRole::WORKER)) {
        return 0;
    }
    std::set<uint32_t> table_set;
    for (auto& executor : _executors) {
        const auto& table_accessors = executor->table_accessors();
        for (auto& itr : table_accessors) {
            table_set.insert(itr.first);
        }
    }
    int ret_size = 0;
    auto table_num = table_set.size();
    std::future<int> rets[table_num];
    for (auto table_id : table_set) {
        rets[ret_size++] = save_model(epoch_id, table_id, way); 
    }
    int all_ret = 0;
    for (int i = 0; i < ret_size; ++i) {
        rets[i].wait();
        all_ret |= rets[i].get();
    }
    return all_ret;
}

int LearnerProcess::load_model(uint64_t epoch_id) {
    auto* environment = _context_ptr->environment.get();
    if (!environment->is_master_node(EnvironmentRole::WORKER)) {
        return 0;
    }
    std::set<uint32_t> loaded_table_set;
    auto model_dir = _context_ptr->epoch_accessor->checkpoint_path();
    for (auto& executor : _executors) {
        const auto& table_accessors = executor->table_accessors();
        for (auto& itr : table_accessors) {
            if (loaded_table_set.count(itr.first)) {
                continue;
            }
            auto table_model_path = _context_ptr->file_system->path_join(
                model_dir, string::format_string("%03d", itr.first));
            if (_context_ptr->file_system->list(table_model_path).size() == 0) {
                VLOG(2) << "miss table_model:" << table_model_path << ", initialize by default";
                auto scope = std::move(executor->fetch_scope());
                CHECK(itr.second[0]->create(scope.get()) == 0);
            } else {
                auto status = _context_ptr->ps_client()->load(itr.first, 
                    model_dir, std::to_string((int)ModelSaveWay::ModelSaveTrainCheckpoint));
                CHECK(status.get() == 0) << "table load failed, id:" << itr.first;
            }
            loaded_table_set.insert(itr.first);
        }
    }
    return 0;
}

int LearnerProcess::run() {
    auto* dataset = _context_ptr->dataset.get();
    auto* environment = _context_ptr->environment.get();
    auto* epoch_accessor = _context_ptr->epoch_accessor.get(); 
    uint64_t epoch_id = epoch_accessor->current_epoch_id();

    environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogLevel::NOTICE, 
        "Resume train with epoch_id:%d %s", epoch_id, _context_ptr->epoch_accessor->text(epoch_id).c_str());
    
    //尝试加载模型 or 初始化
    CHECK(load_model(epoch_id) == 0);
    environment->barrier(EnvironmentRole::WORKER); 

    //判断是否先dump出base
    wait_save_model(epoch_id, ModelSaveWay::ModelSaveInferenceBase);
    environment->barrier(EnvironmentRole::WORKER); 
    
    while (true) {
        epoch_accessor->next_epoch();
        bool already_dump_inference_model = false;
        epoch_id = epoch_accessor->current_epoch_id();
        std::string epoch_log_title = paddle::string::format_string(
            "train epoch_id:%d label:%s", epoch_id, epoch_accessor->text(epoch_id).c_str());
        std::string data_path = paddle::string::to_string<std::string>(dataset->epoch_data_path(epoch_id));
        
        //Step1. 等待样本ready
        {
            environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogLevel::NOTICE, 
                "%s, wait data ready:%s", epoch_log_title.c_str(), data_path.c_str());
            while (dataset->epoch_data_status(epoch_id) != DatasetStatus::Ready) {
                sleep(30);  
                dataset->pre_detect_data(epoch_id);
                environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogLevel::NOTICE, 
                "data not ready, wait 30s");
            } 
            environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogLevel::NOTICE, 
                "Start %s, data is ready", epoch_log_title.c_str());
            environment->barrier(EnvironmentRole::WORKER); 
        }
    
        //Step2. 运行训练网络
        {
            std::map<std::string, paddle::framework::Channel<DataItem>> backup_input_map;
            for (auto& executor : _executors) {
                environment->barrier(EnvironmentRole::WORKER); 
                VLOG(2) << "Start executor:" << executor->train_exe_name();
                auto data_name = executor->train_data_name();
                paddle::framework::Channel<DataItem> input_channel;
                if (backup_input_map.count(data_name)) {
                    input_channel = backup_input_map[data_name];
                } else {
                    input_channel = dataset->fetch_data(data_name, epoch_id);
                }
                input_channel = executor->run(input_channel, dataset->data_parser(data_name));
                VLOG(2) << "End executor:" << executor->train_exe_name();

                // 等待异步梯度完成
                _context_ptr->ps_client()->flush();
                environment->barrier(EnvironmentRole::WORKER); 

                if (executor->is_dump_all_model()) {
                    already_dump_inference_model = true;
                    wait_save_model(epoch_id, ModelSaveWay::ModelSaveInferenceDelta);
                }
                backup_input_map[data_name] = input_channel;
                environment->barrier(EnvironmentRole::WORKER); 
            }
        }

        //Step3. Dump Model For Delta&&Checkpoint
        {
            if (!already_dump_inference_model) {
                already_dump_inference_model = true;
                wait_save_model(epoch_id, ModelSaveWay::ModelSaveInferenceDelta);
            } 
            wait_save_model(epoch_id, ModelSaveWay::ModelSaveTrainCheckpoint);
            environment->barrier(EnvironmentRole::WORKER); 

            epoch_accessor->epoch_done(epoch_id);
            environment->barrier(EnvironmentRole::WORKER); 

        }
    
        //Step4. Output Monitor && RunStatus
        //TODO
    }
    
    return 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
