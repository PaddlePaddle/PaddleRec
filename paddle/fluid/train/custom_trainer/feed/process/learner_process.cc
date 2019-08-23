/*
 *Author: xiexionghang
 *Train样本
 */
#include <omp.h>
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
    if (_context_ptr->epoch_accessor->need_save_model(epoch_id, way)) {
        LOG(NOTICE) << "save table, table_id:" << table_id;
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
    int ret_size = 0;
    auto table_num = _context_ptr->params_table_list.size();
    std::future<int> rets[table_num];
    for (int i = 0; i < table_num; ++i) {
        auto table_id = _context_ptr->params_table_list[i].table_id();
        rets[ret_size++] = save_model(epoch_id, table_id, way); 
    }

    int all_ret = 0;
    for (int i = 0; i < ret_size; ++i) {
        rets[i].wait();
        all_ret |= rets[i].get();
    }
    return all_ret;
}

int LearnerProcess::run() {
    auto* dataset = _context_ptr->dataset.get();
    auto* environment = _context_ptr->environment.get();
    auto* epoch_accessor = _context_ptr->epoch_accessor.get(); 
    uint64_t epoch_id = epoch_accessor->current_epoch_id();

    environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogLevel::NOTICE, 
        "Resume traine with epoch_id:%d label:%s", epoch_id, _context_ptr->epoch_accessor->text(epoch_id).c_str());
    
    //判断是否先dump出base
    wait_save_model(epoch_id, ModelSaveWay::ModelSaveInferenceBase);
    environment->barrier(EnvironmentRole::WORKER); 
    
    while (true) {
        epoch_accessor->next_epoch();
        bool already_dump_inference_model = false;
        epoch_id = epoch_accessor->current_epoch_id();
        std::string epoch_log_title= paddle::string::format_string(
            "train epoch_id:%d label:%s", epoch_id, epoch_accessor->text(epoch_id).c_str());
        
        //Step1. 等待样本ready
        {
            environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogLevel::NOTICE, 
                "Start %s, wait data ready", epoch_log_title.c_str());
            while (dataset->epoch_data_status(epoch_id) != DatasetStatus::Ready) {
                sleep(30);  
                dataset->pre_detect_data(epoch_id);
                environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogLevel::NOTICE, 
                    "%s, data not ready, wait 30s", epoch_log_title.c_str());
            } 
            environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogLevel::NOTICE, 
                "%s, data is ready, start traning", epoch_log_title.c_str());
            environment->barrier(EnvironmentRole::WORKER); 
        }
    
        //Step2. 运行训练网络
        {
            std::map<std::string, paddle::framework::Channel<DataItem>> backup_input_map;
            for (auto& executor : _executors) {
                environment->barrier(EnvironmentRole::WORKER); 
                auto data_name = executor->train_data_name();
                paddle::framework::Channel<DataItem> input_channel;
                if (backup_input_map.count(data_name)) {
                    input_channel = backup_input_map[data_name];
                } else {
                    input_channel = dataset->fetch_data(data_name, epoch_id);
                }
                input_channel = executor->run(input_channel, dataset->data_parser(data_name));
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
        }
    
        //Step4. Output Monitor && RunStatus
        //TODO
    }
    
    return 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
