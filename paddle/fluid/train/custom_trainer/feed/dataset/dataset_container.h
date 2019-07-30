/* DatasetContainer
 * 保存一个数据源的样本，并驱动样本的异步加载
 */
#pragma once
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/train/custom_trainer/feed/common/runtime_environment.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

//单条样本的原始数据
class DataItem {
public:
    DataItem() {}
    virtual ~DataItem() {}
    std::string id;  //样本id标识，可用于shuffle
    std::string data;//样本完整数据
};

class DatasetContainer {
public:
    DatasetContainer() {}
    virtual ~DatasetContainer() {}
    virtual int initialize(const YAML::Node& config) {
        _dataset_config = config;
        _prefetch_num = config["prefetch_num"].as<int>();
        _data_root_path = config["root_path"].as<std::string>();
        _data_path_generater = config["_data_path_generater"].as<std::string>();
        return 0;
    }  
    virtual void run();
    //获取特定epoch_i样本，如果数据未ready，Channel内为空指针
    virtual ::paddle::framework::Channel<DataItem> fetch(int epoch_id);
    //触发可预取的数据判断
    virtual void pre_detect_data(RuntimeEnvironment* env);
    
protected:
    //异步样本download
    virtual void async_download_data();
    virtual void download(int epoch_id, const std::vector<std::string>& paths);
   
    int _prefetch_num = 0;
    YAML::Node _dataset_config;
    std::string _data_root_path;
    std::string _data_path_generater;
    
    uint32_t _current_dataset_idx;             //当前样本数据idx
    int _current_epoch_id = -1;  
    int _ready_epoch_id = -1; //已下载完成的epoch_id
    std::vector<std::shared_ptr<::paddle::framework::Dataset>> _dataset_list;
};

}//namespace feed
}//namespace custom_trainer
}//namespace paddle
