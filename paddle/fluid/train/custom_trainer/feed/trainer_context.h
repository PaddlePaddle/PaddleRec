#pragma once
#include <string>
#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/train/custom_trainer/feed/common/runtime_environment.h"


namespace paddle {
namespace custom_trainer {
namespace feed {

class Process;
class Dataset;
class FileSystem;
class EpochAccessor;

enum class ModelSaveWay {
    ModelSaveTrainCheckpoint = 0,
    ModelSaveInferenceDelta = 1,
    ModelSaveInferenceBase = 2
};

class TableMeta {
public:
    TableMeta() {}
    ~TableMeta() {}
    int table_id() {
        return _id;
    }
private:
    int _id;
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
YAML::Node trainer_config;
paddle::platform::CPUPlace cpu_place;

std::shared_ptr<Dataset> dataset;                          //训练样本
std::shared_ptr<FileSystem> file_system;                   //文件操作辅助类
std::vector<TableMeta> params_table_list;                  //参数表
std::shared_ptr<EpochAccessor> epoch_accessor;             //训练轮次控制
std::shared_ptr<RuntimeEnvironment> environment;           //运行环境
std::vector<std::shared_ptr<Process>> process_list;        //训练流程
std::shared_ptr<SignCacheDict> cache_dict;                 //大模型cache词典
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
