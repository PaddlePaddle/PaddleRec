#pragma once
#include <string>
#include <cmath> //std::lround
#include "paddle/fluid/train/custom_trainer/feed/monitor/monitor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

// cost time profile
class CostMonitor : public Monitor {
public:
    CostMonitor() : _total_time_ms(0), _total_cnt(0), _avg_time_ms(0), _compute_interval(0) {}
    virtual ~CostMonitor() {}

    virtual int initialize(const YAML::Node& config,
        std::shared_ptr<TrainerContext> context_ptr) override;

    //添加一项记录，统计内容Monitor自行从Executor按需获取
    virtual void add_data(int epoch_id, 
            const MultiThreadExecutor* executor, 
            SampleInstance* samples, 
            size_t num);
    
    //是否开始结果统计
    virtual bool need_compute_result(int epoch_id);
    //统计当前结果
    virtual void compute_result() {
        CHECK(_total_cnt != 0);
        _avg_time_ms = _total_time_ms / _total_cnt;
    }
    //基于现有结果，输出格式化的统计信息
    virtual std::string format_result() {
        return paddle::string::format_string(
                "Monitor %s: Cost Time=%lu", Monitor::_name.c_str(), _avg_time_ms);
    }
    
    virtual void reset() {
        _total_time_ms = 0;
        _total_cnt = 0;
        _avg_time_ms = 0;
    }

protected:
    std::string _name;

private:
    uint64_t _total_time_ms;
    uint64_t _total_cnt;
    uint64_t _avg_time_ms;
    uint32_t  _compute_interval;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
