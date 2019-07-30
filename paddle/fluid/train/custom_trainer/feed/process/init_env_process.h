/*
 *Author: xiexionghang
 *用于训练环境的整体配置读取、环境初始化工作
 */
#pragma once
#include "paddle/fluid/train/custom_trainer/feed/process/process.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class InitEnvProcess : public Process {
public:
    InitEnvProcess() {}
    virtual ~InitEnvProcess() {}
    virtual int initialize(std::shared_ptr<TrainerContext> context_ptr);
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
