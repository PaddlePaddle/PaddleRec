#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/epoch_accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {
    int EpochAccessor::initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
        _model_root_path =  config["model_root_path"].as<std::string>();

        _trainer_context = context_ptr.get();
        if (context_ptr->file_system == nullptr) {
            VLOG(0) << "file_system is not initialized";
            return -1;
        }
        auto fs = _trainer_context->file_system.get();
        if (config["donefile"]) {
            _done_file_path = fs->path_join(_model_root_path, config["donefile"].as<std::string>());
        } else {
            _done_file_path = fs->path_join(_model_root_path, "epoch_donefile.txt");
        }
        
        if (!fs->exists(_done_file_path)) {
            VLOG(0) << "missing done file, path:" << _done_file_path;
            return -1;
        }

        std::string done_text = fs->tail(_done_file_path);
        _done_status = paddle::string::split_string(done_text, std::string("\t"));
        _current_epoch_id = get_status<uint64_t>(EpochStatusFiled::EpochIdField);
        _last_checkpoint_epoch_id = get_status<uint64_t>(EpochStatusFiled::CheckpointIdField);
        _last_checkpoint_path = get_status<std::string>(EpochStatusFiled::CheckpointPathField);
        return 0;
    }
    
    int32_t EpochAccessor::epoch_done(uint64_t epoch_id) {
        struct timeval now; 
        gettimeofday(&now, NULL); 
        if (need_save_model(epoch_id, ModelSaveWay::ModelSaveTrainCheckpoint)) {
            _last_checkpoint_epoch_id = epoch_id;
            _last_checkpoint_path = model_save_path(epoch_id, ModelSaveWay::ModelSaveTrainCheckpoint);
        }
        set_status(EpochStatusFiled::EpochIdField, epoch_id);
        set_status(EpochStatusFiled::TimestampField, now.tv_sec);
        set_status(EpochStatusFiled::CheckpointIdField, _last_checkpoint_epoch_id);
        set_status(EpochStatusFiled::CheckpointPathField, _last_checkpoint_path);
        set_status(EpochStatusFiled::DateField, format_timestamp(epoch_id, "%Y%m%d"));

        // 非主节点不做状态持久化
        if (!_trainer_context->environment->is_master_node(EnvironmentRole::WORKER)) {
            return 0;
        }
        auto fs = _trainer_context->file_system.get();
        std::string done_str = paddle::string::join_strings(_done_status, '\t');
        // 保留末尾1000数据
        std::string tail_done_info = paddle::string::trim_spaces(fs->tail(_done_file_path, 1000)); 
        if (tail_done_info.size() > 0) {
            tail_done_info = tail_done_info + "\n" + done_str;
        } else {
            tail_done_info = done_str;
        }
        VLOG(2) << "Write epoch donefile to " << _done_file_path << ", str:" << done_str;
        bool write_success = false;
        while (true) {
            fs->remove(_done_file_path);
            auto fp = fs->open_write(_done_file_path, "");
            if (fwrite(tail_done_info.c_str(), tail_done_info.length(), 1, &*fp) == 1) {
                break;
            }     
            sleep(10);   
        }
        VLOG(2) << "Write epoch donefile success";
        return 0;
    }

    int TimelyEpochAccessor::initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
        _time_zone_seconds = config["time_zone_seconds"].as<int>();
        _train_time_interval = config["train_time_interval"].as<int>();
        CHECK(_train_time_interval > 0 && (_train_time_interval % SecondsPerMin) == 0);
        _train_num_per_day = SecondsPerDay / _train_time_interval;
        return EpochAccessor::initialize(config, context_ptr); 
    }

    void TimelyEpochAccessor::next_epoch() {
        _current_epoch_id = next_epoch_id(_current_epoch_id);
    }

    std::string TimelyEpochAccessor::text(uint64_t epoch_id) {
        auto delta = delta_id(epoch_id);
        std::string date = format_timestamp(epoch_id, "%Y%m%d%H%M");
        return string::format_string("%s delta-%d", date.c_str(), delta);
    }

    uint64_t TimelyEpochAccessor::next_epoch_id(uint64_t epoch_id) {
        if (epoch_id == 0) {
            struct timeval now; 
            gettimeofday(&now, NULL); 
            // 归整到零点
            return now.tv_sec / SecondsPerDay * SecondsPerDay;
        } 
        return epoch_id + _train_time_interval;
    }

    bool TimelyEpochAccessor::is_last_epoch(uint64_t epoch_id) {
        auto delta = delta_id(epoch_id);
        return delta == _train_num_per_day;
    }
 
    uint64_t TimelyEpochAccessor::epoch_time_interval() {
        return _train_time_interval;
    }

    uint64_t TimelyEpochAccessor::epoch_timestamp(uint64_t epoch_id) {
        return epoch_id;
    }
 
    bool TimelyEpochAccessor::need_save_model(uint64_t epoch_id, ModelSaveWay save_way) {
        if (epoch_id == 0) {
            return false;
        }
        switch (save_way) {
            case ModelSaveWay::ModelSaveInferenceDelta:
                return true;
            case ModelSaveWay::ModelSaveInferenceBase:
                return is_last_epoch(epoch_id);
            case ModelSaveWay::ModelSaveTrainCheckpoint:
                return ((epoch_id / SecondsPerHour) % 8) == 0;
        }
        return false;
    }

    std::string TimelyEpochAccessor::model_save_path(uint64_t epoch_id, ModelSaveWay save_way) {
        int32_t delta = delta_id(epoch_id);
        std::string date = format_timestamp(epoch_id, "%Y%m%d");
        std::string date_with_hour = format_timestamp(epoch_id, "%Y%m%d%H");
        switch (save_way) {
            case ModelSaveWay::ModelSaveInferenceDelta:
                return _trainer_context->file_system->path_join(_model_root_path, 
                    string::format_string("xbox/%s/delta-%d", date.c_str(), delta));
            case ModelSaveWay::ModelSaveInferenceBase:
                return _trainer_context->file_system->path_join(_model_root_path, 
                    string::format_string("xbox/%s/base", date.c_str()));
            case ModelSaveWay::ModelSaveTrainCheckpoint:
                return _trainer_context->file_system->path_join(_model_root_path, 
                    string::format_string("batch_model/%s", date_with_hour.c_str()));
        }
        return "";
    }

    REGIST_CLASS(EpochAccessor, TimelyEpochAccessor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
