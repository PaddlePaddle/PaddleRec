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
    _train_thread_num = config["train_thread_num"].as<int>();
    _threads_executor.resize(_train_thread_num);
    
    if (config["executor"]) {
        _executor_num = config["executor"].size();
        omp_set_num_threads(_train_thread_num);
        #pragma omp parallel for
        for (int i = 0; i < _train_thread_num; ++i) {
            _threads_executor[i].resize(_executor_num);
            for (int e = 0; e < _executor_num; ++e) {
                auto e_class = config["executor"][e]["class"].as<std::string>();
                auto* e_ptr = CREATE_INSTANCE(Executor, e_class);
                _threads_executor[i][e].reset(e_ptr);  
                if (e_ptr->initialize(config["executor"][e], context_ptr) != 0) {
                    ret = -1;
                }
            }
        }
    }
    return 0;
}

std::future<int> LearnerProcess::save_model(uint64_t epoch_id, int table_id, ModelSaveWay way) {
    std::promise<int> p;
    auto ret = p.get_future();
    if (_context_ptr->epoch_accessor->need_save_model(epoch_id, way)) {
        //TODO
        //context_ptr->pslib_client()->save();
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
            for (int i = 0; i < _executor_num; ++i) {
                std::vector<std::shared_ptr<std::thread>> train_threads(_train_thread_num);
                for (int thread_id = 0; thread_id < _train_thread_num; ++thread_id) {
                    train_threads[i].reset(new std::thread([this](int exe_idx, int thread_idx) {
                        auto* executor = _threads_executor[thread_idx][exe_idx].get();
                        run_executor(executor);                      
                    }, i, thread_id));
                }   
                for (int i = 0; i < _train_thread_num; ++i) {
                    train_threads[i]->join();
                }
                environment->barrier(EnvironmentRole::WORKER); 

                if (_threads_executor[0][i]->is_dump_all_model()) {
                    already_dump_inference_model = true;
                    wait_save_model(epoch_id, ModelSaveWay::ModelSaveInferenceDelta);
                }
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

int LearnerProcess::run_executor(Executor* executor) {
    //TODO
    return 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
