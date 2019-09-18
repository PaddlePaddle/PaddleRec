/*
 *Author: xiexionghang
 *Train样本
 */
#include <omp.h>
#include "paddle/fluid/platform/timer.h"
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
    _is_dump_cache_model = config["dump_cache_model"].as<bool>(false);
    _cache_load_converter = config["load_cache_converter"].as<std::string>("");
    _startup_dump_inference_base = config["startup_dump_inference_base"].as<bool>(false);
    if (config["executor"]) {
        _executors.resize(config["executor"].size());
        for (size_t i = 0; i < _executors.size(); ++i) {
            _executors[i].reset(new MultiThreadExecutor());
            CHECK(_executors[i]->initialize(config["executor"][i], context_ptr) == 0);
        }
    }
    return 0;
}

// 更新各节点存储的CacheModel
int LearnerProcess::update_cache_model(uint64_t epoch_id, ModelSaveWay way) {
    auto fs = _context_ptr->file_system;
    auto* ps_client = _context_ptr->pslib->ps_client();
    auto* environment = _context_ptr->environment.get();
    auto* epoch_accessor = _context_ptr->epoch_accessor.get();
    if (!epoch_accessor->need_save_model(epoch_id, way)) {
        return 0;
    }
    auto* ps_param = _context_ptr->pslib->get_param();
    if (_is_dump_cache_model && way == ModelSaveWay::ModelSaveInferenceBase) {
        auto model_dir = epoch_accessor->model_save_path(epoch_id, way);
        auto& table_param = ps_param->server_param().downpour_server_param().downpour_table_param();
        for (auto& param : table_param) {
            if (param.type() != paddle::PS_SPARSE_TABLE) {
                continue;
            }
            auto cache_model_path = fs->path_join(
                model_dir, string::format_string("%03d_cache/", param.table_id()));
            if (!fs->exists(cache_model_path)) {
                continue;
            }
            auto& cache_dict = *(_context_ptr->cache_dict.get());
            cache_dict.clear();
            cache_dict.reserve(_cache_sign_max_num);
            auto cache_file_list = fs->list(fs->path_join(cache_model_path, "part*"));
            for (auto& cache_path : cache_file_list) {
                auto cache_file = fs->open_read(cache_path, _cache_load_converter);
                char *buffer = nullptr;
                size_t buffer_size = 0;
                ssize_t line_len = 0;
                while ((line_len = getline(&buffer, &buffer_size, cache_file.get())) != -1) {
                    if (line_len <= 1) {
                        continue;
                    }
                    char* data_ptr = NULL;
                    cache_dict.append(strtoul(buffer, &data_ptr, 10));
                }
                if (buffer != nullptr) {
                    free(buffer);
                } 
            }
            break;
        }
    }
    return 0;
}
int LearnerProcess::wait_save_model(uint64_t epoch_id, ModelSaveWay way, bool is_force_dump) {
    auto fs = _context_ptr->file_system;
    auto* ps_client = _context_ptr->pslib->ps_client();
    auto* environment = _context_ptr->environment.get();
    auto* epoch_accessor = _context_ptr->epoch_accessor.get();
    if (!environment->is_master_node(EnvironmentRole::WORKER)) {
        return 0;
    }
    if (!is_force_dump && !epoch_accessor->need_save_model(epoch_id, way)) {
        return 0;
    }
    paddle::platform::Timer timer;
    timer.Start();
    std::set<uint32_t> table_set;
    auto model_dir = epoch_accessor->model_save_path(epoch_id, way);
    for (auto& executor : _executors) {
        const auto& table_accessors = executor->table_accessors();
        for (auto& itr : table_accessors) {
            table_set.insert(itr.first);
        }
        auto save_path = fs->path_join(model_dir, executor->train_exe_name() + "_param");
        VLOG(2) << "Start save model, save_path:" << save_path;
        executor->save_persistables(save_path);
    }
    int ret_size = 0;
    auto table_num = table_set.size();
    std::future<int> rets[table_num];
    for (auto table_id : table_set) {
        VLOG(2) << "Start save model, table_id:" << table_id;
        rets[ret_size++] = ps_client->save(table_id, model_dir, std::to_string((int)way));
    }
    int all_ret = 0;
    for (int i = 0; i < ret_size; ++i) {
        rets[i].wait();
        all_ret |= rets[i].get();
    }
    timer.Pause();
    VLOG(2) << "Save Model Cost(s):" << timer.ElapsedSec();
    
    // save cache model, 只有inference需要cache_model
    auto* ps_param = _context_ptr->pslib->get_param();
    if (_is_dump_cache_model && (way == ModelSaveWay::ModelSaveInferenceBase ||
        way == ModelSaveWay::ModelSaveInferenceDelta)) {
        auto& table_param = ps_param->server_param().downpour_server_param().downpour_table_param();
        for (auto& param : table_param) {
            if (param.type() != paddle::PS_SPARSE_TABLE) {
                continue;
            } 
            double cache_threshold = 0.0;
            auto status = ps_client->get_cache_threshold(param.table_id(), cache_threshold);
            CHECK(status.get() == 0) << "CacheThreshold Get failed!";
            status = ps_client->cache_shuffle(param.table_id(), model_dir, std::to_string((int)way),
                std::to_string(cache_threshold));
            CHECK(status.get() == 0) << "Cache Shuffler Failed";
            status = ps_client->save_cache(param.table_id(), model_dir, std::to_string((int)way));
            auto feature_size = status.get();
            CHECK(feature_size >= 0) << "Cache Save Failed";
            auto cache_model_path = fs->path_join(
                model_dir, string::format_string("%03d_cache/sparse_cache.meta", param.table_id()));
            auto cache_meta_file = fs->open_write(cache_model_path, "");
            auto meta = string::format_string("file_prefix:part\npart_num:%d\nkey_num:%d\n", 
                param.sparse_table_cache_file_num(), feature_size);
            CHECK(fwrite(meta.c_str(), meta.size(), 1, cache_meta_file.get()) == 1) << "Cache Meta Failed";
            if (feature_size > _cache_sign_max_num) {
                _cache_sign_max_num = feature_size;
            }
        }
    }
    _context_ptr->epoch_accessor->update_model_donefile(epoch_id, way);

    return all_ret;
}

int LearnerProcess::load_model(uint64_t epoch_id) {
    auto* environment = _context_ptr->environment.get();
    if (!environment->is_master_node(EnvironmentRole::WORKER)) {
        return 0;
    }
    auto* fs = _context_ptr->file_system.get();
    std::set<uint32_t> loaded_table_set;
    auto model_dir = _context_ptr->epoch_accessor->checkpoint_path();
    paddle::platform::Timer timer;
    timer.Start();
    for (auto& executor : _executors) {
        const auto& table_accessors = executor->table_accessors();
        for (auto& itr : table_accessors) {
            if (loaded_table_set.count(itr.first)) {
                continue;
            }
            auto table_model_path = fs->path_join(
                model_dir, string::format_string("%03d", itr.first));
            if ((!fs->exists(table_model_path)) || fs->list(table_model_path).size() == 0) {
                VLOG(2) << "miss table_model:" << table_model_path << ", initialize by default";
                auto scope = std::move(executor->fetch_scope());
                CHECK(itr.second[0]->create(scope.get()) == 0);
            } else {
                ENVLOG_WORKER_MASTER_NOTICE("Loading model %s", model_dir.c_str());
                auto status = _context_ptr->ps_client()->load(itr.first, 
                    model_dir, std::to_string((int)ModelSaveWay::ModelSaveTrainCheckpoint));
                CHECK(status.get() == 0) << "table load failed, id:" << itr.first;
            }
            loaded_table_set.insert(itr.first);
        }
    }
    timer.Pause();
    ENVLOG_WORKER_MASTER_NOTICE("Finished loading model, cost:%f", timer.ElapsedSec());
    return 0;
}

int LearnerProcess::run() {
    auto* dataset = _context_ptr->dataset.get();
    auto* environment = _context_ptr->environment.get();
    auto* epoch_accessor = _context_ptr->epoch_accessor.get(); 
    uint64_t epoch_id = epoch_accessor->current_epoch_id();

    ENVLOG_WORKER_MASTER_NOTICE("Resume train with epoch_id:%d %s", epoch_id, _context_ptr->epoch_accessor->text(epoch_id).c_str());
    //尝试加载模型 or 初始化
    CHECK(load_model(epoch_id) == 0);
    environment->barrier(EnvironmentRole::WORKER); 

    //判断是否先dump出base TODO
    wait_save_model(epoch_id, ModelSaveWay::ModelSaveInferenceBase, _startup_dump_inference_base);
    environment->barrier(EnvironmentRole::WORKER); 
    
    while (true) {
        epoch_accessor->next_epoch();
        _context_ptr->monitor_ssm.str(""); 
        bool already_dump_inference_model = false;
        epoch_id = epoch_accessor->current_epoch_id();
        std::string epoch_log_title = paddle::string::format_string(
            "train epoch_id:%d label:%s", epoch_id, epoch_accessor->text(epoch_id).c_str());
        std::string data_path = paddle::string::to_string<std::string>(dataset->epoch_data_path(epoch_id));
        ENVLOG_WORKER_MASTER_NOTICE("    ==== begin %s ====", epoch_accessor->text(epoch_id).c_str());
        //Step1. 等待样本ready
        {
            ENVLOG_WORKER_MASTER_NOTICE("      %s, wait data ready:%s", epoch_log_title.c_str(), data_path.c_str());
            while (dataset->epoch_data_status(epoch_id) != DatasetStatus::Ready) {
                sleep(30);  
                dataset->pre_detect_data(epoch_id);
                ENVLOG_WORKER_MASTER_NOTICE("      epoch_id:%d data not ready, wait 30s", epoch_id);
            } 
            ENVLOG_WORKER_MASTER_NOTICE("      Start %s, data is ready", epoch_log_title.c_str());
            environment->barrier(EnvironmentRole::WORKER); 
        }
    
        //Step2. 运行训练网络
        {
            std::map<std::string, paddle::framework::Channel<DataItem>> backup_input_map;
            for (auto& executor : _executors) {
                environment->barrier(EnvironmentRole::WORKER); 
                paddle::platform::Timer timer;
                timer.Start();
                ENVLOG_WORKER_MASTER_NOTICE("Start executor:%s", executor->train_exe_name().c_str());
                auto data_name = executor->train_data_name();
                paddle::framework::Channel<DataItem> input_channel;
                if (backup_input_map.count(data_name)) {
                    input_channel = backup_input_map[data_name];
                } else {
                    input_channel = dataset->fetch_data(data_name, epoch_id);
                }
                input_channel = executor->run(input_channel, dataset->data_parser(data_name));
                timer.Pause();
                ENVLOG_WORKER_MASTER_NOTICE("End executor:%s, cost:%f", executor->train_exe_name().c_str(), timer.ElapsedSec());

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
            wait_save_model(epoch_id, ModelSaveWay::ModelSaveInferenceBase);
            environment->barrier(EnvironmentRole::WORKER); 
            update_cache_model(epoch_id, ModelSaveWay::ModelSaveInferenceBase);
            environment->barrier(EnvironmentRole::WORKER); 

            if (epoch_accessor->is_last_epoch(epoch_id)) {
                wait_save_model(epoch_id, ModelSaveWay::ModelSaveTrainCheckpointBase);
            } else {
                wait_save_model(epoch_id, ModelSaveWay::ModelSaveTrainCheckpoint);
            }
            environment->barrier(EnvironmentRole::WORKER); 
            if (epoch_accessor->is_last_epoch(epoch_id) &&
                environment->is_master_node(EnvironmentRole::WORKER)) {
                paddle::platform::Timer timer;
                timer.Start();
                ENVLOG_WORKER_MASTER_NOTICE("Start shrink table");
                for (auto& executor : _executors) {
                    const auto& table_accessors = executor->table_accessors();
                    for (auto& itr : table_accessors) {
                        CHECK(itr.second[0]->shrink() == 0);
                    }
                } 
                timer.Pause();
                ENVLOG_WORKER_MASTER_NOTICE("End shrink table, cost:%f", timer.ElapsedSec());
            }
            environment->barrier(EnvironmentRole::WORKER); 

            epoch_accessor->epoch_done(epoch_id);
            environment->barrier(EnvironmentRole::WORKER); 
        }
        ENVLOG_WORKER_MASTER_NOTICE("    ==== end %s ====", epoch_accessor->text(epoch_id).c_str());
        //Step4. Output Monitor && RunStatus
        //TODO
    }
    
    return 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
