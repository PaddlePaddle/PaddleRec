/*
 *Author: xiexionghang
 *运行环境，屏蔽MPI or Local环境的运行差异
 *为了兼容不同环境的底层实现，Env的接口调用条件严格于sum(limit(env[n]))
 *如：MPI环境下，写接口只允许单线程调用，那么默认对所有Env保证此调用限制
 */
#pragma once
#include <yaml-cpp/yaml.h>
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

enum class EnvironmentLogLevel {
    FATAL       = 0,
    ERROR       = 1,
    NOTICE      = 2,
    DEBUG       = 3
};

enum class EnvironmentLogType {
    MASTER_LOG      = 0, //仅master节点对外输出
    ALL_LOG         = 1 //所有节点都会对外输出
};

class RuntimeEnvironment {
public:
    RuntimeEnvironment() {}
    virtual ~RuntimeEnvironment() {}
    //配置初始化
    virtual int initialize(YAML::Node config) = 0;
    //环境初始化，会在所有依赖模块initialize后调用
    virtual int wireup() = 0;
    
    //多线程可调用接口  Start
    //当前环境rank_idx
    virtual uint32_t rank_idx() = 0;
    //环境内主节点
    virtual bool is_master_node() {
        return rank_idx() == 0;
    }
    //环境定制化log
    template<class... ARGS>
    void log(EnvironmentLogType type, EnvironmentLogLevel level,
        const char* fmt, ARGS && ... args) {
        print_log(type, level, paddle::string::format_string(fmt, args...));
    }
    //多线程可调用接口      End


    //接口只允许在主线程调用   Start
    //barrier
    virtual void barrier_all() = 0;
    //接口只允许在主线程调用   End
protected:
    virtual void print_log(EnvironmentLogType type, EnvironmentLogLevel level,  const std::string& log_str) = 0;
};
REGISTER_REGISTERER(RuntimeEnvironment);

class MPIRuntimeEnvironment : public RuntimeEnvironment {
public:
    MPIRuntimeEnvironment() {}
    virtual ~MPIRuntimeEnvironment() {}
    //配置初始化
    virtual int initialize(YAML::Node config);
    //环境初始化，会在所有依赖模块initialize后调用
    virtual int wireup();
    //当前环境rank_idx
    virtual uint32_t rank_idx();

    virtual void barrier_all();
protected:
    virtual void print_log(EnvironmentLogType type, EnvironmentLogLevel level,  const std::string& log_str);
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
