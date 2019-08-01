/*
 *Author: xiexionghang
 *用于训练环境的整体配置读取、环境初始化工作
 */
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/epoch_accessor.h"
#include "paddle/fluid/train/custom_trainer/feed/process/init_env_process.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int InitEnvProcess::initialize(std::shared_ptr<TrainerContext> context_ptr) {
    Process::initialize(context_ptr);
    paddle::framework::InitDevices(false);
    context_ptr->cpu_place = paddle::platform::CPUPlace();
    
    YAML::Node config = _context_ptr->trainer_config;
    //environment
    std::string env_class = config["environment"]["environment_class"].as<std::string>();
    auto* environment = CREATE_CLASS(RuntimeEnvironment, env_class);
    if (environment->initialize(config["environment"]) != 0) {
        return -1;
    }
    context_ptr->environment.reset(environment);

    //epoch
    std::string epoch_class = config["epoch"]["epoch_class"].as<std::string>();
    auto* epoch = CREATE_CLASS(EpochAccessor, epoch_class);
    if (epoch->initialize(config["epoch"], context_ptr) != 0) {
        return -1;
    }
    context_ptr->epoch_accessor.reset(epoch);
    VLOG(3) << "Env initialize success"; 
    return 0;
}

int InitEnvProcess::run() {
    //step 1. psserver init
    //step2. psserver load
    return 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
