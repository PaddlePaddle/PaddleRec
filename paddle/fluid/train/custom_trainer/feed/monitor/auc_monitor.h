#pragma once
#include <string>
#include "paddle/fluid/train/custom_trainer/feed/monitor/monitor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

//TODO 完善AucMonitor 

class AucMonitor : public Monitor {
public:
    AucMonitor() {}
    virtual ~AucMonitor() {}

    virtual int initialize(const YAML::Node& config,
        std::shared_ptr<TrainerContext> context_ptr) {
        Monitor::initialize(config, context_ptr);
        //一些额外配置 对于AUC主要是target && label 信息
        return 0;
    }

    //添加一项记录，统计内容Monitor自行从Executor按需获取
    virtual void add_data(int epoch_id, const Executor* executor);
    
    //是否开始结果统计
    virtual bool need_compute_result(int epoch_id, EpochAccessor* accessor);
    //统计当前结果
    virtual void compute_result();
    //基于现有结果，输出格式化的统计信息
    virtual std::string format_result();
    
    virtual void reset();
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
