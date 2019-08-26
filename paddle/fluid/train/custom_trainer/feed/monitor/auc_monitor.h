#pragma once
#include <string>
#include <cmath> //std::lround
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
        std::shared_ptr<TrainerContext> context_ptr) override;

    //添加一项记录，统计内容Monitor自行从Executor按需获取
    virtual void add_data(int epoch_id, 
            const Executor* executor, 
            SampleInstance* instance, 
            size_t num);
    
    //是否开始结果统计
    virtual bool need_compute_result(int epoch_id, EpochAccessor* accessor);
    //统计当前结果
    virtual void compute_result();
    //基于现有结果，输出格式化的统计信息
    virtual std::string format_result();
    
    virtual void reset();

protected:
    std::string _label_name;
    std::string _target_name;
    std::string _name;
    std::string _output_var;
    std::mutex _mutex;
    double _local_abserr, _local_sqrerr, _local_pred;
    double _auc;
    double _mae;
    double _rmse;
    double _actual_ctr, _predicted_ctr;
    double _size;
    double _bucket_error;
    int _table_size;
    void add_unlocked(double pred, int label);

private:
    void calculate_bucket_error();
    void set_table_size(int table_size);

    uint32_t  _compute_interval;
    std::vector<double> _table[2];
    static constexpr double kRelativeErrorBound = 0.05;
    static constexpr double kMaxSpan = 0.01;    
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
