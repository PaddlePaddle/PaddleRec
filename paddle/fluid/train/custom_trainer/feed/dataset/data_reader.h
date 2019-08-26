/* DataReader
 * 对指定数据的读取
 */
#pragma once
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/train/custom_trainer/feed/common/pipeline.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class TrainerContext;

struct FeatureItem {
public:
    FeatureItem() {
    }
    FeatureItem(uint64_t sign_, uint16_t slot_) {
        sign() = sign_;
        slot() = slot_;
    }
    uint64_t& sign() {
        return *(uint64_t*)sign_buffer();
    }
    const uint64_t& sign() const {
        return *(const uint64_t*)sign_buffer();
    }
    uint16_t& slot() {
        return _slot;
    }
    const uint16_t& slot() const {
        return _slot;
    }

private:
    char _sign[sizeof(uint64_t)];
    uint16_t _slot;

    char* sign_buffer() const {
        return (char*)_sign;
    }
};

struct SampleInstance {
    std::string id;
    std::vector<float> predicts;
    std::vector<float> labels;
    std::vector<FeatureItem> features;
    std::vector<float> embedx;
};

class DataItem {
public:
    DataItem() {}
    virtual ~DataItem() {}
    std::string id;  //样本id标识，可用于shuffle
    std::string data;//样本数据， maybe压缩格式
};

typedef std::shared_ptr<Pipeline<DataItem, SampleInstance>> SampleInstancePipe;
inline SampleInstancePipe make_sample_instance_channel() {
    return std::make_shared<Pipeline<DataItem, SampleInstance>>();
}

class DataParser {
public:
    DataParser() {}
    virtual ~DataParser() {}
    virtual int initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context) = 0;
    virtual int parse(const std::string& str, DataItem& data) const {
        return parse(str.c_str(), data);
    }
    virtual int parse(const char* str, size_t len, DataItem& data) const = 0;
    virtual int parse(const char* str, DataItem& data) const = 0;
    virtual int parse_to_sample(const DataItem& data, SampleInstance& instance) const = 0;  
};
REGISTER_REGISTERER(DataParser);

class DataReader {
public:
    DataReader() {}
    virtual ~DataReader() {}
    virtual int initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context);
    //判断样本数据是否已就绪，就绪表明可以开始download
    virtual bool is_data_ready(const std::string& data_dir) = 0;
    //读取dir下文件列表
    virtual std::vector<std::string> data_file_list(const std::string& data_dir) = 0;
    //读取目录下数据到样本流中
    virtual int read_all(const std::string& data_dir, ::paddle::framework::Channel<DataItem> data_channel) = 0;
    //读取指定文件列表的数据到样本流中
    virtual int read_all(const std::vector<std::string>& data_list, ::paddle::framework::Channel<DataItem> data_channel) = 0;
    virtual const DataParser* get_parser() {
        return _parser.get();
    }
protected:
    std::shared_ptr<DataParser> _parser;//数据格式转换
    std::string _pipeline_cmd; //将文件流，重定向到pipeline_cmd，再读入
};
REGISTER_REGISTERER(DataReader);

}//namespace feed
}//namespace custom_trainer
}//namespace paddle
