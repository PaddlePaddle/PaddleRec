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

// 保持该枚举值的连续递增，且ALL在尾部
enum class EnvironmentRole {
    WORKER          = 0,        //训练Worker
    PSERVER         = 1,        //参数服务器

    ALL             = 2         //所有角色，请保持在枚举尾部
};

// Reduce的操作类型
enum class ReduceOperator {
    SUM             = 0         //求和
};

class RuntimeEnvironment {
public:
    RuntimeEnvironment();
    virtual ~RuntimeEnvironment();
    // 配置初始化
    virtual int initialize(YAML::Node config) = 0;

    // job 信息
    virtual std::string job_id() {
        return _job_id;
    }
    virtual std::string job_name() {
        return _job_name;
    }

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
    // 全局reduce操作, 返回reduce结果
    virtual double all_reduce(double x, ReduceOperator op, EnvironmentRole role) {
        double result = x;
        all_reduce_in_place(&result, 1, op, role);
        return result;
    }
    // 全局reduce，就地执行
    virtual void all_reduce_in_place(double* x, int n, 
            ReduceOperator op, EnvironmentRole role) = 0;
    // 接口只允许在主线程调用   End
protected:
    virtual void print_log(EnvironmentRole role, EnvironmentLogType type, 
        EnvironmentLogLevel level,  const std::string& log_str) = 0;

    std::string _debug_verion;
    std::string _job_id = "default_job_id";
    std::string _job_name = "default_job_name";
};
REGIST_REGISTERER(RuntimeEnvironment);

#define ENVLOG_WORKER_ALL_NOTICE \
environment->log(EnvironmentRole::WORKER, EnvironmentLogType::ALL_LOG, EnvironmentLogType::NOTICE, 
#define ENVLOG_WORKER_MASTER_NOTICE \
environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogType::NOTICE, 
#define ENVLOG_WORKER_ALL_ERROR \
environment->log(EnvironmentRole::WORKER, EnvironmentLogType::ALL_LOG, EnvironmentLogType::ERROR, 
#define ENVLOG_WORKER_MASTER_ERROR \
environment->log(EnvironmentRole::WORKER, EnvironmentLogType::MASTER_LOG, EnvironmentLogType::ERROR, 

std::string format_timestamp(time_t time, const char* format);
inline std::string format_timestamp(time_t time, const std::string& format) {
    return format_timestamp(time, format.c_str());
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
