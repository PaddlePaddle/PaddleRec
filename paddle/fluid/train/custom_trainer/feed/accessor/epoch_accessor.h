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
    
    virtual uint64_t current_epoch_id() {
        return _current_epoch_id;
    }
    virtual void next_epoch() = 0;
    virtual std::string text(uint64_t epoch_id) = 0;
    virtual bool data_ready(uint64_t epoch_id) = 0;
    virtual int next_epoch_id(uint64_t epoch_id) = 0;
    virtual bool is_last_epoch(uint64_t epoch_id) = 0; 
    //epoch间的数据时间间隔（秒）
    virtual uint64_t epoch_time_interval() = 0;
    //获取epoch的样本数据时间
    virtual uint64_t epoch_timestamp(uint64_t epoch_id) = 0; 
    virtual bool need_save_model(uint64_t epoch_id, ModelSaveWay save_way) = 0;
    virtual std::string model_save_path(uint64_t epoch_id, ModelSaveWay save_way) = 0;
protected:
    uint64_t _current_epoch_id;
};
REGISTER_REGISTERER(EpochAccessor);

class HourlyEpochAccessor : public EpochAccessor {
public:
    HourlyEpochAccessor() {}
    virtual ~HourlyEpochAccessor() {}
    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr);
    virtual void next_epoch();
    virtual std::string text(uint64_t epoch_id);
    virtual bool data_ready(uint64_t epoch_id);
    virtual int next_epoch_id(uint64_t epoch_id);
    virtual bool is_last_epoch(uint64_t epoch_id);
    virtual uint64_t epoch_time_interval();
    virtual uint64_t epoch_timestamp(uint64_t epoch_id); 
    virtual bool need_save_model(uint64_t epoch_id, ModelSaveWay save_way);
    virtual std::string model_save_path(uint64_t epoch_id, ModelSaveWay save_way);
private:
    std::string _model_root_path;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
