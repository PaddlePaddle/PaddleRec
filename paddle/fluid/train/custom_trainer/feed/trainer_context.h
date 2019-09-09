#pragma once
#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/train/custom_trainer/feed/common/yaml_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/common/pslib_warpper.h"
#include "paddle/fluid/train/custom_trainer/feed/common/runtime_environment.h"


namespace paddle {
namespace custom_trainer {
namespace feed {

class PSlib;
class Process;
class Dataset;
class FileSystem;
class EpochAccessor;

const uint32_t SecondsPerMin = 60;
const uint32_t SecondsPerHour = 3600;
const uint32_t SecondsPerDay = 24 * 3600;

enum class ModelSaveWay {
    ModelSaveTrainCheckpoint = 0,
    ModelSaveInferenceDelta = 1,
    ModelSaveInferenceBase = 2
};

enum class TrainerStatus {
    Training  = 0,  // 训练状态
    Saving    = 1  // 模型存储状态
};

class SignCacheDict {
public:
    int32_t sign2index(uint64_t sign) {
        return -1;
    }

    uint64_t index2sign(int32_t index) {
        return 0;
    }
};

class TrainerContext {
public:
    inline paddle::ps::PSClient* ps_client() {
        return pslib->ps_client();
    }
    inline bool is_status(TrainerStatus status) {
        auto bit_idx = static_cast<uint32_t>(status);
        return ((trainer_status >> bit_idx) & 1) > 0;
    }
    // 非线程安全, 其实TrainerContext所有成员的线程安全性 取决于 成员本身的线程安全性
    inline void set_status(TrainerStatus status, bool on) {
        auto bit_idx = static_cast<uint32_t>(status);
        trainer_status = trainer_status & (1L << bit_idx);
    }

    uint32_t trainer_status;      // trainer当前，由于可同时处于多种状态，这里分bit存储状态
    YAML::Node trainer_config;
    paddle::platform::CPUPlace cpu_place;

    std::shared_ptr<PSlib> pslib;
    std::stringstream monitor_ssm;                             //记录monitor信息
    std::shared_ptr<Dataset> dataset;                          //训练样本
    std::shared_ptr<FileSystem> file_system;                   //文件操作辅助类
    std::shared_ptr<EpochAccessor> epoch_accessor;             //训练轮次控制
    std::shared_ptr<RuntimeEnvironment> environment;           //运行环境
    std::vector<std::shared_ptr<Process>> process_list;        //训练流程
    std::shared_ptr<SignCacheDict> cache_dict;                 //大模型cache词典
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
