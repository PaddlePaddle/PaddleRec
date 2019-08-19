#pragma once
#include <boost/lexical_cast.hpp>
#include "paddle/fluid/string/to_string.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/accessor.h"
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

enum class EpochStatusFiled {
    DateField                = 0,
    TimestampField           = 1,
    CheckpointPathField      = 2,
    EpochIdField             = 3,
    CheckpointIdField        = 4
};

class EpochAccessor : public Accessor {
public:
    EpochAccessor() {}
    virtual ~EpochAccessor() {}
    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr);
    
    virtual uint64_t current_epoch_id() {
        return _current_epoch_id;
    }

    virtual const std::string& checkpoint_path() {
        return _last_checkpoint_path;
    }
    
    template <class T>
    T get_status(EpochStatusFiled field) {
        auto status = paddle::string::trim_spaces(_done_status[static_cast<int>(field)]);
        return boost::lexical_cast<T>(status.c_str());
    }
    
    virtual void next_epoch()                     = 0;
    virtual std::string model_root_path() {
        return _model_root_path;
    }
    virtual std::string text(uint64_t epoch_id)   = 0;
    virtual uint64_t next_epoch_id(uint64_t epoch_id)  = 0;
    virtual bool is_last_epoch(uint64_t epoch_id) = 0; 
    //epoch间的数据时间间隔（秒）
    virtual uint64_t epoch_time_interval() = 0;
    //获取epoch的样本数据时间
    virtual uint64_t epoch_timestamp(uint64_t epoch_id) = 0; 
    virtual bool need_save_model(uint64_t epoch_id, ModelSaveWay save_way) = 0;
    virtual std::string model_save_path(uint64_t epoch_id, ModelSaveWay save_way) = 0;
protected:
    TrainerContext* _trainer_context;
    std::string _done_file_path;
    std::string _model_root_path;
    uint64_t _current_epoch_id = 0;
    std::string _last_checkpoint_path;
    uint64_t _last_checkpoint_epoch_id = 0;
    std::vector<std::string> _done_status; //当前完成状态，统一存成string
    
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
    virtual uint64_t next_epoch_id(uint64_t epoch_id);
    virtual bool is_last_epoch(uint64_t epoch_id);
    virtual uint64_t epoch_time_interval();
    virtual uint64_t epoch_timestamp(uint64_t epoch_id); 
    virtual bool need_save_model(uint64_t epoch_id, ModelSaveWay save_way);
    virtual std::string model_save_path(uint64_t epoch_id, ModelSaveWay save_way);
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
