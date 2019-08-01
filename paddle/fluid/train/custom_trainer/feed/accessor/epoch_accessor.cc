#pragma once
#include "paddle/fluid/train/custom_trainer/feed/accessor/epoch_accessor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {
    int HourlyEpochAccessor::initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) {
        return 0;
    }
    void HourlyEpochAccessor::next_epoch() {
        _current_epoch_id = next_epoch_id(_current_epoch_id);
    }
    std::string HourlyEpochAccessor::text(int epoch_id) {
        return std::to_string(epoch_id);
    }
    bool HourlyEpochAccessor::data_ready(int epoch_id) {
        return true;
    }
    int HourlyEpochAccessor::next_epoch_id(int epoch_id) {
        if (epoch_id <= 0) {
            struct timeval now; 
            gettimeofday(&now, NULL); 
            return now.tv_sec / (24 * 3600) * (24 * 3600);
        } 
        return epoch_id + 3600;
    }
    bool HourlyEpochAccessor::is_last_epoch(int epoch_id) {
        return ((epoch_id / 3600) % 24) == 23;
    } 
    bool HourlyEpochAccessor::need_save_model(int epoch_id, ModelSaveWay save_way) {
        if (epoch_id <= 0) {
            return false;
        }
        if (save_way == ModelSaveWay::ModelSaveInferenceDelta) {
            return true;
        } else if (save_way == ModelSaveWay::ModelSaveInferenceBase) {
            return is_last_epoch(epoch_id);
        } else if (save_way == ModelSaveWay::ModelSaveTrainCheckpoint) {
            return ((epoch_id / 3600) % 8) == 0;
        }
        return false;
    }
    std::string HourlyEpochAccessor::model_save_path(int epoch_id, ModelSaveWay save_way) {
        if (save_way == ModelSaveWay::ModelSaveInferenceDelta) {
            return _model_root_path + "/xbox/delta-" + std::to_string(epoch_id);
        } else if (save_way == ModelSaveWay::ModelSaveInferenceBase) {
            return _model_root_path + "/xbox/base";
        } else if (save_way == ModelSaveWay::ModelSaveTrainCheckpoint) {
            return _model_root_path + "/xbox/checkpoint";
        }
        return "";
    }
    REGISTER_CLASS(EpochAccessor, HourlyEpochAccessor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
