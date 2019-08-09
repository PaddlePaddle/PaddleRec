/*
 *Author: xiexionghang
 *Train样本
 */
#pragma once
#include "paddle/fluid/train/custom_trainer/feed/process/process.h"
#include "paddle/fluid/train/custom_trainer/feed/executor/executor.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

typedef std::vector<std::shared_ptr<Executor>> MultiExecutor;
class LearnerProcess : public Process {
public:
    LearnerProcess() {}
    virtual ~LearnerProcess() {}
    
    virtual int run();
    virtual int initialize(std::shared_ptr<TrainerContext> context_ptr);

protected:
//同步保存所有模型
virtual int wait_save_model(uint64_t epoch_id, ModelSaveWay way);
//异步保存指定模型
virtual std::future<int> save_model(uint64_t epoch_id, int table_id, ModelSaveWay way);
//执行指定训练网络
virtual int run_executor(Executor* executor);



private:
    int _executor_num = 0;    //需要执行训练的网络个数
    int _train_thread_num = 1;//并行训练线程数
    std::vector<MultiExecutor> _threads_executor;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
