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
#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"
#include "paddle/fluid/train/custom_trainer/feed/common/runtime_environment.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class Shuffler;

inline int data_num_for_train(uint64_t train_begin_timestamp, uint32_t train_time_interval, uint32_t data_time_interval) {
    uint64_t data_begin_time = train_begin_timestamp;
    uint64_t data_end_time = data_begin_time + train_time_interval;
    uint64_t end_idx = (data_end_time + data_time_interval - 1) / data_time_interval;
    uint64_t begin_idx = (data_begin_time + data_time_interval - 1 ) / data_time_interval; 
    return end_idx - begin_idx;
}

enum class DatasetStatus {
    Empty          = 0,
    Detected       = 1,
    Downloding     = 2,
    Ready          = 3
};

struct DatasetInfo {
    uint64_t timestamp = 0;
    std::vector<std::string> file_path_list;
    DatasetStatus status = DatasetStatus::Empty;
    ::paddle::framework::Channel<DataItem> data_channel = ::paddle::framework::MakeChannel<DataItem>();
};

class DatasetContainer {
public:
    DatasetContainer() {}
    virtual ~DatasetContainer() {
        if (_downloader_thread != nullptr) {
            _stop_download = true;
            _downloader_thread->join();
        }
    }
    virtual int initialize(
        const YAML::Node& config, std::shared_ptr<TrainerContext> context);
    // 触发可预取的数据判断
    virtual void pre_detect_data(uint64_t epoch_id);
    // 获取epoch对应的样本数据目录
    std::vector<std::string> epoch_data_path(uint64_t epoch_id);
    // 获取数据状态
    virtual DatasetStatus epoch_data_status(uint64_t epoch_id);
    // 获取特定epoch_i样本，如果数据未ready，Channel内为空指针
    virtual ::paddle::framework::Channel<DataItem> fetch(uint64_t epoch_id);
    // 获取DataItem解析器
    virtual const DataParser* data_parser() {
        return _data_reader->get_parser();
    }
protected:
    virtual DatasetStatus data_status(uint64_t timestamp);
    virtual int read_data_list(const std::string& data_dir, std::vector<std::string>& data_list);
    // 异步样本download
    virtual void async_download_data(uint64_t start_timestamp);
    virtual std::shared_ptr<DatasetInfo> dataset(uint64_t timestamp);
   
    int _prefetch_num                  = 0;
    bool _stop_download                = false;
    int _data_split_interval           = 60;                 //样本切分周期(秒)
    YAML::Node _dataset_config;
    std::string _data_path_formater;
    std::vector<std::string> _data_root_paths;              //支持同时读取多个目录
    
    TrainerContext* _trainer_context;
    std::shared_ptr<Shuffler> _shuffler;
    std::shared_ptr<DataReader> _data_reader;
    std::shared_ptr<std::thread> _downloader_thread;
    std::vector<std::shared_ptr<DatasetInfo>> _dataset_list;//预取的数据列表
};

}//namespace feed
}//namespace custom_trainer
}//namespace paddle
