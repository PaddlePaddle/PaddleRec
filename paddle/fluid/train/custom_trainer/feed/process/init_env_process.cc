/*
 *Author: xiexionghang
 *用于训练环境的整体配置读取、环境初始化工作
 */
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/train/custom_trainer/feed/process/init_env_process.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int InitEnvProcess::initialize(std::shared_ptr<TrainerContext> context_ptr) {
    paddle::framework::InitDevices(false);
    context_ptr->cpu_place = paddle::platform::CPUPlace();
    VLOG(3) << "Env initialize success"; 
    return 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
