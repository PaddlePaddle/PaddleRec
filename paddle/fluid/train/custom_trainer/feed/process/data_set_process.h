/*
 *Author: xiexionghang
 *组织训练样本的读取工作
 */
#pragma once
#include "paddle/fluid/train/custom_trainer/feed/process/process.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class DatasetProcess : public Process {
public:
    DatasetProcess() {}
    virtual ~DatasetProcess() {}
    virtual int initialize(std::shared_ptr<TrainerContext> context_ptr);
private:
    std::map<std::string, DatasetContainer> _dataset_map;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
