#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"

#include <cstdio>

#include <glog/logging.h>

#include "paddle/fluid/framework/io/fs.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class LineDataParser : public DataParser{
public:
    LineDataParser() {}

    virtual ~LineDataParser() {}

    virtual int initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context) {
        return 0;
    }

    virtual int parse(const char* str, size_t len, DataItem& data) const {
        size_t pos = 0;
        while (pos < len && str[pos] != ' ') {
            ++pos;
        }
        if (pos >= len) {
            VLOG(2) << "fail to parse line: " << std::string(str, len) << ", strlen: " << len;
            return -1;
        }
        VLOG(5) << "getline: "  << str << " , pos: " << pos << ", len: " << len;
        data.id.assign(str, pos);
        data.data.assign(str + pos + 1, len - pos - 1);
        if (!data.data.empty() && data.data.back() == '\n') {
            data.data.pop_back();
        }
        return 0;
    }

    virtual int parse(const char* str, DataItem& data) const {
        size_t pos = 0;
        while (str[pos] != '\0' && str[pos] != ' ') {
            ++pos;
        }
        if (str[pos] == '\0') {
            VLOG(2) << "fail to parse line: " << str << ", get '\\0' at pos: " << pos;
            return -1;
        }
        VLOG(5) << "getline: "  << str << " , pos: " << pos;
        data.id.assign(str, pos);
        data.data.assign(str + pos + 1);
        if (!data.data.empty() && data.data.back() == '\n') {
            data.data.pop_back();
        }
        return 0;
    }

    virtual int parse_to_sample(const DataItem& data, SampleInstance& instance) const {
        return 0;
    }
};
REGISTER_CLASS(DataParser, LineDataParser);

int DataReader::initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context) {
    _parser.reset(CREATE_CLASS(DataParser, config["parser"]["class"].as<std::string>()));
    if (_parser == nullptr) {
        VLOG(2) << "fail to get parser: " << config["parser"]["class"].as<std::string>();
        return -1;
    }
    if (_parser->initialize(config["parser"], context) != 0) {
        VLOG(2) << "fail to initialize parser" << config["parser"]["class"].as<std::string>();
        return -1;
    }
    _pipeline_cmd = config["pipeline_cmd"].as<std::string>();
    return 0;
}

class LineDataReader : public DataReader {
public:
    LineDataReader() {}
    virtual ~LineDataReader() {}
    virtual int initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context) {
        if (DataReader::initialize(config, context) != 0) {
            return -1;
        }
        _done_file_name = config["done_file"].as<std::string>();
        _buffer_size = config["buffer_size"].as<int>(1024);
        _filename_prefix = config["filename_prefix"].as<std::string>("");
        _buffer.reset(new char[_buffer_size]);
        return 0;
    }

    //判断样本数据是否已就绪，就绪表明可以开始download
    virtual bool is_data_ready(const std::string& data_dir) {
        auto done_file_path = framework::fs_path_join(data_dir, _done_file_name);
        if (framework::fs_exists(done_file_path)) {
            return true;
        }
        return false;
    }

    virtual std::vector<std::string> data_file_list(const std::string& data_dir) {
        if (_filename_prefix.empty()) {
            return framework::fs_list(data_dir);
        }
        std::vector<std::string> data_files;
        for (auto& filepath : framework::fs_list(data_dir)) {
            auto filename = framework::fs_path_split(filepath).second;
            if (filename.size() >= _filename_prefix.size() && filename.substr(0, _filename_prefix.size()) == _filename_prefix) {
                data_files.push_back(std::move(filepath));
            }
        }
        return data_files;
    }

    //读取数据样本流中
    virtual int read_all(const std::string& data_dir, framework::Channel<DataItem> data_channel) {
        framework::ChannelWriter<DataItem> writer(data_channel.get());
        DataItem data_item;
        if (_buffer_size <= 0 || _buffer == nullptr) {
            VLOG(2) << "no buffer";
            return -1;
        }
        for (const auto& filepath : data_file_list(data_dir)) {
            if (framework::fs_path_split(filepath).second == _done_file_name) {
                continue;
            }
            int err_no = 0;
            std::shared_ptr<FILE> fin = framework::fs_open_read(filepath, &err_no, _pipeline_cmd);
            if (err_no != 0) {
                VLOG(2) << "fail to open file: " << filepath << ", with cmd: " << _pipeline_cmd;
                return -1;
            }
            while (fgets(_buffer.get(), _buffer_size, fin.get())) {
                if (_parser->parse(_buffer.get(), data_item) != 0) {
                    return -1;
                }
                writer << std::move(data_item);
            }
            if (ferror(fin.get()) != 0) {
                VLOG(2) << "fail to read file: " << filepath;
                return -1;
            }
        }
        writer.Flush();
        if (!writer) {
            VLOG(2) << "fail when write to channel";
            return -1;
        }
        data_channel->Close();
        return 0;
    }

    virtual const DataParser* get_parser() {
        return _parser.get();
    }
private:
    std::string _done_file_name; // without data_dir
    int _buffer_size = 0;
    std::unique_ptr<char[]> _buffer;
    std::string _filename_prefix;
};
REGISTER_CLASS(DataReader, LineDataReader);

}//namespace feed
}//namespace custom_trainer
}//namespace paddle
