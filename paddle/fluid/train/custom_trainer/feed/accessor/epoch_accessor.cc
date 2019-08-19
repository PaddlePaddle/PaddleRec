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
        
        if (config["donefile"]) {
            _done_file_path = _trainer_context->file_system->path_join(_model_root_path, config["donefile"].as<std::string>());
        } else {
            _done_file_path = _trainer_context->file_system->path_join(_model_root_path, "epoch_donefile.txt");
        }
        
        if (!_trainer_context->file_system->exists(_done_file_path)) {
            VLOG(0) << "missing done file, path:" << _done_file_path;
        }

        std::string done_text = _trainer_context->file_system->tail(_done_file_path);
        _done_status = paddle::string::split_string(done_text, std::string("\t"));
        _current_epoch_id = get_status<uint64_t>(EpochStatusFiled::EpochIdField);
        _last_checkpoint_epoch_id = get_status<uint64_t>(EpochStatusFiled::CheckpointIdField);
        _last_checkpoint_path = get_status<std::string>(EpochStatusFiled::CheckpointPathField);
        return 0;
    }

    int HourlyEpochAccessor::initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
        EpochAccessor::initialize(config, context_ptr); 
        return 0;
    }

    void HourlyEpochAccessor::next_epoch() {
        _current_epoch_id = next_epoch_id(_current_epoch_id);
    }

    std::string HourlyEpochAccessor::text(uint64_t epoch_id) {
        return format_timestamp(epoch_id, "%Y%m%d delta-%H");
    }

    uint64_t HourlyEpochAccessor::next_epoch_id(uint64_t epoch_id) {
        if (epoch_id == 0) {
            struct timeval now; 
            gettimeofday(&now, NULL); 
            return now.tv_sec / (24 * 3600) * (24 * 3600);
        } 
        return epoch_id + 3600;
    }

    bool HourlyEpochAccessor::is_last_epoch(uint64_t epoch_id) {
        return ((epoch_id / 3600) % 24) == 23;
    }
 
    uint64_t HourlyEpochAccessor::epoch_time_interval() {
        return 3600;
    }

    uint64_t HourlyEpochAccessor::epoch_timestamp(uint64_t epoch_id) {
        return epoch_id;
    }
 
    bool HourlyEpochAccessor::need_save_model(uint64_t epoch_id, ModelSaveWay save_way) {
        if (epoch_id == 0) {
            return false;
        }
        switch (save_way) {
            case ModelSaveWay::ModelSaveInferenceDelta:
                return true;
            case ModelSaveWay::ModelSaveInferenceBase:
                return is_last_epoch(epoch_id);
            case ModelSaveWay::ModelSaveTrainCheckpoint:
                return ((epoch_id / 3600) % 8) == 0;
        }
        return false;
    }

    std::string HourlyEpochAccessor::model_save_path(uint64_t epoch_id, ModelSaveWay save_way) {
        switch (save_way) {
            case ModelSaveWay::ModelSaveInferenceDelta:
                return _trainer_context->file_system->path_join(_model_root_path, "/xbox/delta-" + std::to_string(epoch_id));
            case ModelSaveWay::ModelSaveInferenceBase:
                return _trainer_context->file_system->path_join(_model_root_path, "/xbox/base");
            case ModelSaveWay::ModelSaveTrainCheckpoint:
                return _trainer_context->file_system->path_join(_model_root_path, "/xbox/checkpoint");
        }
        return "";
    }

    REGISTER_CLASS(EpochAccessor, HourlyEpochAccessor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
