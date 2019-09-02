#include "paddle/fluid/train/custom_trainer/feed/monitor/auc_monitor.h"
#include "paddle/fluid/train/custom_trainer/feed/executor/multi_thread_executor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int AucMonitor::initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context_ptr) {
    Monitor::initialize(config, context_ptr);
    _target_idx = config["target_idx"].as<int32_t>();
    _target_name = config["target"].as<std::string>();
    _table_size = 1000000;
    if (config["table_size"]) {
        _table_size = config["table_size"].as<int>();
    }
    set_table_size(_table_size);
    _compute_interval = config["compute_interval"].as<uint32_t>();
    CHECK(_compute_interval % 60 == 0);
    return 0;
}

void AucMonitor::add_data(int epoch_id, 
    const MultiThreadExecutor* executor, ScopeExecutorContext* ctx) {
    auto num = ctx->sample_num();
    auto* samples = ctx->samples();
    CHECK(num > 0);
    std::lock_guard<std::mutex> lock(_mutex);
    for (int i = 0; i < num; ++i) {
        auto& instance = samples[i]; 
        add_unlocked(instance.predicts[_target_idx], std::lround(instance.labels[_target_idx]));
    }
}

bool AucMonitor::need_compute_result(int epoch_id) {
    CHECK(_epoch_accessor != nullptr);
    uint64_t epoch_time = _epoch_accessor->epoch_timestamp(epoch_id);
    return epoch_time % _compute_interval == 0;
}
void AucMonitor::compute_result() {
    auto* environment = Monitor::_context_ptr->environment.get();
    double* table[2] = {&_table[0][0], &_table[1][0]};
    for (int i = 0; i < 2; i++) {
        environment->all_reduce_in_place(table[i], 
            _table_size, ReduceOperator::SUM, EnvironmentRole::WORKER);
    }
    double area = 0;
    double fp = 0;
    double tp = 0;
    for (int i = _table_size - 1; i >= 0; i--) {
        double newfp = fp + table[0][i];
        double newtp = tp + table[1][i];
        area += (newfp - fp) * (tp + newtp) / 2;
        fp = newfp;
        tp = newtp;
    }
    _auc = area / (fp * tp);
    _mae = environment->all_reduce(_local_abserr, 
        ReduceOperator::SUM, EnvironmentRole::WORKER) / (fp + tp);
    _rmse = sqrt(environment->all_reduce(_local_sqrerr,
        ReduceOperator::SUM, EnvironmentRole::WORKER) / (fp + tp));
    _actual_ctr = tp / (fp + tp);
    _predicted_ctr = environment->all_reduce(_local_pred,
        ReduceOperator::SUM, EnvironmentRole::WORKER) / (fp + tp);
    _size = fp + tp;
    calculate_bucket_error();
}

std::string AucMonitor::format_result() {
    double copc = 0.0;
    if (fabs(_predicted_ctr) > 1e-6) {
        copc = _actual_ctr / _predicted_ctr;
    }
    return paddle::string::format_string("AUC=%.6f BUCKET_ERROR=%.6f MAE=%.6f RMSE=%.6f "
        "Actual CTR=%.6f Predicted CTR=%.6f COPC=%.6f INS Count=%.0f",
        _auc,
        _bucket_error,
        _mae, 
        _rmse, 
        _actual_ctr, 
        _predicted_ctr, 
        copc,
        _size);
}

void AucMonitor::add_unlocked(double pred, int label) {
    if (std::isnan(pred)) {
        VLOG(2) << "pred[" << pred << "] outside of [0,1]";
        return;
    }
    CHECK(pred >= 0 && pred <= 1) << "pred[" << pred << "] outside of [0,1]";
    CHECK(label == 0 || label == 1) << "label[" << label << "] invalid";
    _table[label][std::min(int(pred * _table_size), _table_size - 1)]++;
    _local_abserr += fabs(pred - label);
    _local_sqrerr += (pred - label) * (pred - label);
    _local_pred += pred;
}

void AucMonitor::calculate_bucket_error() {
    double last_ctr = -1;
    double impression_sum = 0;
    double ctr_sum = 0.0;
    double click_sum = 0.0;
    double error_sum = 0.0;
    double error_count = 0;
    double* table[2] = {&_table[0][0], &_table[1][0]};
    for (int i = 0; i < _table_size; i++) {
        double click = table[1][i];
        double show = table[0][i] + table[1][i];
        double ctr = (double)i / _table_size;
        if (fabs(ctr - last_ctr) > kMaxSpan) {
            last_ctr = ctr;
            impression_sum = 0.0;
            ctr_sum = 0.0;
            click_sum = 0.0;
        }
        impression_sum += show;
        ctr_sum += ctr * show;
        click_sum += click;
        double adjust_ctr = ctr_sum / impression_sum;
        double relative_error = sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum));
        if (relative_error < kRelativeErrorBound) {
            double actual_ctr = click_sum / impression_sum;
            double relative_ctr_error = fabs(actual_ctr / adjust_ctr - 1);
            error_sum += relative_ctr_error * impression_sum;
            error_count += impression_sum;
            last_ctr = -1;
        }
    }
    _bucket_error = error_count > 0 ? error_sum / error_count : 0.0;
}

void AucMonitor::set_table_size(int table_size) {
    CHECK(table_size >= 1);
    _table_size = table_size;
    for (int i = 0; i < 2; i++) {
        _table[i] = std::vector<double>();
    }
    reset();
}

void AucMonitor::reset() {
    for (int i = 0; i < 2; i++) {
        _table[i].assign(_table_size, 0.0);
    }
    _local_abserr = 0;
    _local_sqrerr = 0;
    _local_pred = 0;
}

REGIST_CLASS(Monitor, AucMonitor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
