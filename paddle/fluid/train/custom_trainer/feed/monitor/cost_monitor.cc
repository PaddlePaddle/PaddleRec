#include "paddle/fluid/train/custom_trainer/feed/monitor/cost_monitor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int CostMonitor::initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context_ptr) {
    Monitor::initialize(config, context_ptr);
    _compute_interval = 3600;
    if (config["compute_interval"]) {
        uint32_t interval = config["compute_interval"].as<uint32_t>();
        if (interval != 3600 || interval != 86400) {
            LOG(FATAL) << " AucMonitor config compute_interval just support hour: 3600 or day: 86400. ";
            return -1;
        }
        _compute_interval = interval;
    }
}

void CostMonitor::add_data(int epoch_id,
        const Executor* executor,
        SampleInstance* instance,
        size_t num) {
    CHECK(executor != nullptr);
    _total_time_ms += executor->epoch_cost();
    _total_cnt ++;
}

bool CostMonitor::need_compute_result(int epoch_id, EpochAccessor* accessor) {
    CHECK(accessor != nullptr);
    uint64_t epoch_time = accessor->epoch_timestamp(epoch_id);
    CHECK(_compute_interval != 0);
    if (epoch_time % _compute_interval != 0) {
        return false;
    }
    return true;
}

std::string CostMonitor::format_result() {
    char buf[1024];
    snprintf(buf, 1024 * sizeof(char), "%s: Cost Time=%lu", 
            Monitor::_name.c_str(),
            _avg_time_ms);
    return std::string(buf);
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
