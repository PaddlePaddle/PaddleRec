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
// 同步保存所有模型
virtual int wait_save_model(uint64_t epoch_id, ModelSaveWay way);

private:
    std::vector<std::shared_ptr<MultiThreadExecutor>> _executors;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
