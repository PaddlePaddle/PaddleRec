#pragma once
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"
#include "paddle/fluid/train/custom_trainer/feed/dataset/dataset_container.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class Dataset {
public:
    Dataset() {}
    virtual ~Dataset() {}
    
    virtual int initialize(
        const YAML::Node& config, std::shared_ptr<TrainerContext> context);

    //触发可预取的数据判断
    virtual void pre_detect_data(const std::string& data_name, uint64_t epoch_id);

    //获取数据状态
    virtual DatasetStatus epoch_data_status(const std::string& data_name, uint64_t epoch_id);

    //返回各DataContainer内的原始数据(maybe 压缩格式)
    virtual ::paddle::framework::Channel<DataItem> fetch_data(
            const std::string& data_name, uint64_t epoch_id);

    //以管道形式返回标准样本流，管道内会对数据做异步转换
    virtual SampleInstancePipe fetch_sample(
            const std::string& data_name, uint32_t batch_size, uint64_t epoch_id);
     
private: 
    std::unordered_map<std::string, std::shared_ptr<DatasetContainer>> _data_containers;
};

} // namespace feed
} // namespace custom_trainer
} // namespace paddle
