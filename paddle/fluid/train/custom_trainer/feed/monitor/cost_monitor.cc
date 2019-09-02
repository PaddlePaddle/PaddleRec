#include "paddle/fluid/train/custom_trainer/feed/monitor/cost_monitor.h"
#include "paddle/fluid/train/custom_trainer/feed/executor/multi_thread_executor.h"

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
    const MultiThreadExecutor* executor, ScopeExecutorContext* ctx) {
    auto num = ctx->sample_num();
    auto* samples = ctx->samples();
    CHECK(executor != nullptr);
    //TODO use paddle time
    _total_time_ms += 1;
    _total_cnt ++;
}

bool CostMonitor::need_compute_result(int epoch_id) {
    uint64_t epoch_time = _epoch_accessor->epoch_timestamp(epoch_id);
    return epoch_time % _compute_interval == 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
