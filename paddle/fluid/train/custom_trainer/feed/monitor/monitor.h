#pragma once
#include <string>
#include "paddle/fluid/train/custom_trainer/feed/accessor/epoch_accessor.h"
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"
#include "paddle/fluid/train/custom_trainer/feed/executor/executor.h"
#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class Monitor {
public:
    Monitor() {}
    virtual ~Monitor() {}

    virtual int initialize(const YAML::Node& config,
        std::shared_ptr<TrainerContext> context_ptr) {
        _name = config["name"].as<std::string>();
        _context_ptr = context_ptr;
        return 0;
    }

    //添加一项记录，统计内容Monitor自行从Executor按需获取
    virtual void add_data(int epoch_id, const Executor* executor, SampleInstance* instance, size_t num) = 0;
    
    //是否对于当前epoch_id进行结果统计
    virtual bool need_compute_result(int epoch_id, EpochAccessor* accessor) = 0;
    //统计当前结果
    virtual void compute_result() = 0;
    //基于现有结果，输出格式化的统计信息
    virtual std::string format_result() = 0;
    
    virtual void reset() = 0;

    const std::string& get_name() {
        return _name;
    }

protected:
    std::string _name;
    std::shared_ptr<TrainerContext> _context_ptr;
};

REGIST_REGISTERER(Monitor);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
