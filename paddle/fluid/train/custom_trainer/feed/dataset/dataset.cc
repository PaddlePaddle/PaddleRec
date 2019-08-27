#include "paddle/fluid/train/custom_trainer/feed/dataset/dataset.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int Dataset::initialize(
    const YAML::Node& config, std::shared_ptr<TrainerContext> context) {
    if (config["data_list"].Type() != YAML::NodeType::Map) {
        LOG(FATAL) << "miss data_list config in dataset, or type error please check";
        return -1;
    }
    for (auto& data_config : config["data_list"]) {
        std::string name = data_config.first.as<std::string>();
        auto data_ptr = std::make_shared<DatasetContainer>();
        if (data_ptr->initialize(data_config.second, context) != 0) {
            LOG(FATAL) << "dataset initialize failed, name:" << name;
            return -1;
        }
        _data_containers[name] = data_ptr;
    }
    return 0;
}

inline void Dataset::pre_detect_data(uint64_t epoch_id) {
    for (auto it = _data_containers.begin(); it != _data_containers.end(); ++it) {
        it->second->pre_detect_data(epoch_id);
    }
    return;
}
inline void Dataset::pre_detect_data(
    const std::string& data_name, uint64_t epoch_id) {
    _data_containers[data_name]->pre_detect_data(epoch_id);
    return;
}

inline DatasetStatus Dataset::epoch_data_status(uint64_t epoch_id) {
    int status = static_cast<int>(DatasetStatus::Ready);
    for (auto it = _data_containers.begin(); it != _data_containers.end(); ++it) {
        auto d_status = static_cast<int>(it->second->epoch_data_status(epoch_id));
        status = d_status < status ? d_status : status;
    }
    return static_cast<DatasetStatus>(status);
}

inline DatasetStatus Dataset::epoch_data_status(
    const std::string& data_name, uint64_t epoch_id) {
    return _data_containers[data_name]->epoch_data_status(epoch_id);
}

inline std::vector<std::string> Dataset::epoch_data_path(
    const std::string& data_name, uint64_t epoch_id) {
    return _data_containers[data_name]->epoch_data_path(epoch_id);
}

inline std::vector<std::string> Dataset::epoch_data_path(uint64_t epoch_id) {
    std::vector<std::string> results;
    for (auto it = _data_containers.begin(); it != _data_containers.end(); ++it) {
        auto items = std::move(it->second->epoch_data_path(epoch_id));
        for (auto& item : items) {
            results.emplace_back(item);
        }
    }
    return results;
}

inline ::paddle::framework::Channel<DataItem> Dataset::fetch_data(
    const std::string& data_name, uint64_t epoch_id) {
    return _data_containers[data_name]->fetch(epoch_id);
}

inline const DataParser* Dataset::data_parser(const std::string& data_name) {
    auto* data_container = _data_containers[data_name].get();
    return data_container->data_parser();
}
     

} // namespace feed
} // namespace custom_trainer
} // namespace paddle
