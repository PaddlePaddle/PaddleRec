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
virtual int update_cache_model(uint64_t epoch_id, ModelSaveWay way);

private:
    bool _is_dump_cache_model;          // 是否进行cache dump
    uint32_t _cache_sign_max_num = 0;   // cache sign最大个数
    std::string _cache_load_converter;  // cache加载的前置转换脚本
    bool _startup_dump_inference_base;  // 启动立即dump base
    std::vector<std::shared_ptr<MultiThreadExecutor>> _executors;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
