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
    virtual void pre_detect_data(uint64_t epoch_id);
    virtual void pre_detect_data(const std::string& data_name, uint64_t epoch_id);

    //获取数据状态
    virtual DatasetStatus epoch_data_status(uint64_t epoch_id);
    virtual DatasetStatus epoch_data_status(const std::string& data_name, uint64_t epoch_id);

    //获取数据路径
    virtual std::vector<std::string> epoch_data_path(uint64_t epoch_id);
    virtual std::vector<std::string> epoch_data_path(const std::string& data_name, uint64_t epoch_id);

    //返回各DataContainer内的原始数据(maybe 压缩格式)
    virtual ::paddle::framework::Channel<DataItem> fetch_data(
            const std::string& data_name, uint64_t epoch_id);

    //获取DataItem解析器
    virtual const DataParser* data_parser(const std::string& data_name);
    
private: 
    std::unordered_map<std::string, std::shared_ptr<DatasetContainer>> _data_containers;
};

} // namespace feed
} // namespace custom_trainer
} // namespace paddle
