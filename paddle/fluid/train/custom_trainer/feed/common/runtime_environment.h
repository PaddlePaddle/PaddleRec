/*
 *Author: xiexionghang
 *运行环境，屏蔽MPI or Local环境的运行差异
 *为了兼容不同环境的底层实现，Env的接口调用条件严格于sum(limit(env[n]))
 *如：MPI环境下，写接口只允许单线程调用，那么默认对所有Env保证此调用限制
 */
#pragma once
#include <yaml-cpp/yaml.h>
#include "communicate/ps_env.h"
#include "paddle/fluid/framework/archive.h"
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
    MASTER_LOG      = 0,        //仅master节点对外输出
    ALL_LOG         = 1         //所有节点都会对外输出
};

//保持该枚举值的连续递增，且ALL在尾部
enum class EnvironmentRole {
    WORKER          = 0,        //训练Worker
    PSERVER         = 1,        //参数服务器

    ALL             = 2         //所有角色，请保持在枚举尾部
};

class RuntimeEnvironment {
public:
    RuntimeEnvironment();
    virtual ~RuntimeEnvironment();
    // 配置初始化
    virtual int initialize(YAML::Node config) = 0;
    // 设置role
    virtual int add_role(EnvironmentRole role) = 0;
    // 判断role
    virtual bool is_role(EnvironmentRole role) = 0;
    // 环境初始化，会在所有依赖模块initialize后调用
    virtual int wireup() = 0;
    
    // 多线程可调用接口  Start
    // 当前环境rank_idx
    virtual uint32_t rank_id(EnvironmentRole role) = 0;
    // 运行环境节点数
    virtual uint32_t node_num(EnvironmentRole role) = 0;
    // 环境内主节点
    virtual bool is_master_node(EnvironmentRole role);
    //For PS
    virtual paddle::ps::PSEnvironment* ps_environment() = 0;
    
    // 环境定制化log
    template<class... ARGS>
    void log(EnvironmentRole role, EnvironmentLogType type, 
        EnvironmentLogLevel level, const char* fmt, ARGS && ... args) {
        print_log(role, type, level, paddle::string::format_string(fmt, args...));
    }
    // 多线程可调用接口      End


    // 接口只允许在主线程调用   Start
    // barrier 指定role的节点
    virtual void barrier(EnvironmentRole role) = 0;
    // bcast 广播
    virtual void bcast(paddle::framework::BinaryArchive& ar, int root_id, EnvironmentRole role) = 0;
    // 接口只允许在主线程调用   End
protected:
    virtual void print_log(EnvironmentRole role, EnvironmentLogType type, 
        EnvironmentLogLevel level,  const std::string& log_str) = 0;
};
REGIST_REGISTERER(RuntimeEnvironment);

std::string format_timestamp(time_t time, const char* format);
inline std::string format_timestamp(time_t time, const std::string& format) {
    return format_timestamp(time, format.c_str());
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
