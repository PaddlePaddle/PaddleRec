/* DatasetContainer
 * 保存一个数据源的样本，并驱动样本的异步加载
 */
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "paddle/fluid/framework/io/shell.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/epoch_accessor.h"
#include "paddle/fluid/train/custom_trainer/feed/dataset/dataset_container.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int DatasetContainer::initialize(
        const YAML::Node& config, std::shared_ptr<TrainerContext> context) {
    _dataset_config = config;
    _trainer_context = context.get();
    //预取n轮样本数据
    _prefetch_num = config["prefetch_num"].as<int>();
    _dataset_list.resize(_prefetch_num);
    for (int i = 0; i < _prefetch_num; ++i) {
        _dataset_list[i].reset(new DatasetInfo);
    }

    _data_root_paths = paddle::string::split_string(
        config["root_path"].as<std::string>(), " ");
    _data_split_interval = config["data_spit_interval"].as<int>();
    _data_path_formater = config["data_path_formater"].as<std::string>();
    std::string data_reader_class = config["data_reader"].as<std::string>();
    DataReader* data_reader = CREATE_CLASS(DataReader, data_reader_class);
    _data_reader.reset(data_reader);
    return _data_reader->initialize(config, context);
}   

std::shared_ptr<DatasetInfo> DatasetContainer::dataset(uint64_t timestamp) {
    auto* epoch_accessor = _trainer_context->epoch_accessor.get();
    auto data_idx = timestamp / epoch_accessor->epoch_time_interval();
    return _dataset_list[data_idx % _prefetch_num];
}

void DatasetContainer::pre_detect_data(uint64_t epoch_id) {
    int status = 0;
    auto* epoch_accessor = _trainer_context->epoch_accessor.get();
    time_t timestamp = epoch_accessor->epoch_timestamp(epoch_id);
    if (timestamp % epoch_accessor->epoch_time_interval() != 0) {
        LOG(FATAL) << "timestamp:" << timestamp << " don't match interval:" << epoch_accessor->epoch_time_interval();
        return;
    }
    if (_downloader_thread == nullptr) {
        _downloader_thread.reset(new std::thread([this, timestamp](){
            async_download_data(timestamp);
        }));
    }
    for (int detect_idx = 0 ; detect_idx < _prefetch_num; ++detect_idx) {
        if (DatasetStatus::Empty != data_status(timestamp)) {
            continue;
        }
        size_t data_num = data_num_for_train(timestamp, epoch_accessor->epoch_time_interval(), _data_split_interval);
        uint64_t data_timestamp = timestamp % _data_split_interval == 0 ? timestamp : (timestamp / _data_split_interval + 1) * _data_split_interval;
        std::vector<std::string> data_path_list;
        for (int i = 0; i < _data_root_paths.size() && status == 0; ++i) {
            for (int j = 0; j < data_num && status == 0; ++j) {
                std::string path_suffix = format_timestamp(data_timestamp + j * _data_split_interval, _data_path_formater);
                std::string data_dir = _data_root_paths[i] + "/" + path_suffix;
                status = read_data_list(data_dir, data_path_list);
            }
        }
        if (status == 0) {
            auto dataset_info = dataset(timestamp);
            dataset_info->timestamp = timestamp;
            dataset_info->file_path_list = std::move(data_path_list);
            dataset_info->status = DatasetStatus::Detected;
        }
        timestamp += epoch_accessor->epoch_time_interval();
    }
    return;
}

int DatasetContainer::read_data_list(const std::string& data_dir, std::vector<std::string>& data_list) {
    auto* environment = _trainer_context->environment.get();
    
    // 检查数据Ready
    int data_status = -1;
    if (environment->is_master_node(EnvironmentRole::WORKER)) {
        if (_data_reader->is_data_ready(data_dir)) {
            data_status = 0;
        }
    }
    paddle::framework::BinaryArchive ar;
    ar << data_status; 
    environment->bcast(ar, 0, EnvironmentRole::WORKER);
    ar >> data_status;
    if (data_status != 0) {
        return -1;
    } 
    
    // 读取文件列表
    ar.Clear();
    std::vector<std::string> data_path_list;
    if (environment->is_master_node(EnvironmentRole::WORKER)) {
         data_path_list = _data_reader->data_file_list(data_dir);
        ar << data_path_list;
    }
    environment->bcast(ar, 0, EnvironmentRole::WORKER);
    ar >> data_path_list;
    auto worker_id = environment->rank_id(EnvironmentRole::WORKER);
    auto worker_num = environment->node_num(EnvironmentRole::WORKER); 
    for (int i = worker_id; i < data_path_list.size(); i+=worker_num) {
        data_list.push_back(data_path_list[i]);
    }
    environment->barrier(EnvironmentRole::WORKER);
    return 0;
}

DatasetStatus DatasetContainer::epoch_data_status(uint64_t epoch_id) {
    auto* epoch_accessor = _trainer_context->epoch_accessor.get();
    time_t timestamp = epoch_accessor->epoch_timestamp(epoch_id);
    return data_status(timestamp);
}

DatasetStatus DatasetContainer::data_status(uint64_t timestamp) {
    auto dataset_info = dataset(timestamp);
    if (dataset_info->timestamp != timestamp) {
        return DatasetStatus::Empty;
    }
    return dataset_info->status;
}
     
paddle::framework::Channel<DataItem> DatasetContainer::fetch(uint64_t epoch_id) {
    paddle::framework::Channel<DataItem> result;
    auto* epoch_accessor = _trainer_context->epoch_accessor.get();
    time_t timestamp = epoch_accessor->epoch_timestamp(epoch_id);
    if (data_status(timestamp) != DatasetStatus::Ready) {
        return result;
    }
    auto dataset_info = dataset(timestamp);
    return dataset_info->data_channel;
}  

void DatasetContainer::async_download_data(uint64_t start_timestamp) {
    auto* epoch_accessor = _trainer_context->epoch_accessor.get();
    if (start_timestamp % epoch_accessor->epoch_time_interval() != 0) {
        LOG(FATAL) << "timestamp:" << start_timestamp << " don't match interval:" << epoch_accessor->epoch_time_interval();
        return;
    }
    while (!_stop_download) {
        auto dataset_info = dataset(start_timestamp);
        while (data_status(start_timestamp) != DatasetStatus::Detected) {
            sleep(30);
        }
        const auto& file_list = dataset_info->file_path_list;
        dataset_info->data_channel->Clear();
        while (_data_reader->read_all(file_list, dataset_info->data_channel) != 0) {
            dataset_info->data_channel->Clear();
            VLOG(0) << "timestamp:" << start_timestamp << " data read failed, retry";
            sleep(30); 
        }
        start_timestamp += epoch_accessor->epoch_time_interval();
    }
}

} // namespace feed
} // namespace custom_trainer
} // namespace paddle
