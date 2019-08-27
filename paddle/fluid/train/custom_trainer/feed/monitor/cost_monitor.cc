#include "paddle/fluid/train/custom_trainer/feed/monitor/cost_monitor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int CostMonitor::initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context_ptr) {
    Monitor::initialize(config, context_ptr);
    if (config["compute_interval"]) {
        _compute_interval = config["compute_interval"].as<uint32_t>();
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

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
