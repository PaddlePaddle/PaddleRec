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

class TrainerContext {
public:
YAML::Node trainer_config;
paddle::platform::CPUPlace cpu_place;
std::vector<TableMeta> params_table_list;
std::shared_ptr<EpochAccessor> epoch_accessor;
std::shared_ptr<RuntimeEnvironment> environment;
std::vector<std::shared_ptr<Process>> process_list;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
