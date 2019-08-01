#pragma once
#include "paddle/fluid/train/custom_trainer/feed/accessor/accessor.h"
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class EpochAccessor : public Accessor {
public:
    EpochAccessor() {}
    virtual ~EpochAccessor() {}
    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) = 0;
    
    virtual int current_epoch_id() {
        return _current_epoch_id;
    }
    virtual void next_epoch() = 0;
    virtual std::string text(int epoch_id) = 0;
    virtual bool data_ready(int epoch_id) = 0;
    virtual int next_epoch_id(int epoch_id) = 0;
    virtual bool is_last_epoch(int epoch_id) = 0; 
    virtual bool need_save_model(int epoch_id, ModelSaveWay save_way) = 0;
    virtual std::string model_save_path(int epoch_id, ModelSaveWay save_way) = 0;
protected:
    int _current_epoch_id;
};
REGISTER_REGISTERER(EpochAccessor);

class HourlyEpochAccessor : public EpochAccessor {
public:
    HourlyEpochAccessor() {}
    virtual ~HourlyEpochAccessor() {}
    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr);
    virtual void next_epoch();
    virtual std::string text(int epoch_id);
    virtual bool data_ready(int epoch_id);
    virtual int next_epoch_id(int epoch_id);
    virtual bool is_last_epoch(int epoch_id);
    virtual bool need_save_model(int epoch_id, ModelSaveWay save_way);
    virtual std::string model_save_path(int epoch_id, ModelSaveWay save_way);
private:
    std::string _model_root_path;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
