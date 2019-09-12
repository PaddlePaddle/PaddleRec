/*
 *Author: xiexionghang
 *Train样本
 */
#pragma once
#include "paddle/fluid/train/custom_trainer/feed/process/process.h"
#include "paddle/fluid/train/custom_trainer/feed/executor/multi_thread_executor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {
class LearnerProcess : public Process {
public:
    LearnerProcess() {}
    virtual ~LearnerProcess() {}
    
    virtual int run();
    virtual int initialize(std::shared_ptr<TrainerContext> context_ptr);

protected:
// 加载所有模型
virtual int load_model(uint64_t epoch_id);
// 同步保存所有模型, is_force_dump:不判断dump条件,强制dump出模型
virtual int wait_save_model(uint64_t epoch_id, ModelSaveWay way, bool is_force_dump = false);

private:
    bool _startup_dump_inference_base;  //启动立即dump base
    std::vector<std::shared_ptr<MultiThreadExecutor>> _executors;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
