#include "paddle/fluid/train/custom_trainer/feed/common/runtime_environment.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

    //配置初始化
    int MPIRuntimeEnvironment::initialize(YAML::Node config) {
        return 0;
    }
    //环境初始化，会在所有依赖模块initialize后调用
    int MPIRuntimeEnvironment::wireup() {
        return 0;
    }
    //当前环境rank_idx
    uint32_t MPIRuntimeEnvironment::rank_idx() {
        return 0;
    }
    void MPIRuntimeEnvironment::barrier_all() {
        return;
    }
    void MPIRuntimeEnvironment::print_log(EnvironmentLogType type, EnvironmentLogLevel level,  const std::string& log_str) {
        if (type ==  EnvironmentLogType::MASTER_LOG && !is_master_node()) {
            return;
        }
        VLOG(2) << log_str;
        return;
    }
    REGISTER_CLASS(RuntimeEnvironment, MPIRuntimeEnvironment);
    

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
