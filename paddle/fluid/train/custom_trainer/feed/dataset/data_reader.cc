#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"

#include <cstdio>

#include <glog/logging.h>
#include <omp.h>

#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class LineDataParser : public DataParser {
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
        VLOG(5) << "getline: " << str << " , pos: " << pos << ", len: " << len;
        data.id.assign(str, pos);
        data.data.assign(str + pos + 1, len - pos - 1);
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
        VLOG(5) << "getline: " << str << " , pos: " << pos;
        data.id.assign(str, pos);
        data.data.assign(str + pos + 1);
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
        _filename_prefix = config["filename_prefix"].as<std::string>("");

        if (config["file_system"] && config["file_system"]["class"]) {
            _file_system.reset(
                    CREATE_CLASS(FileSystem, config["file_system"]["class"].as<std::string>()));
            if (_file_system == nullptr ||
                _file_system->initialize(config["file_system"], context) != 0) {
                VLOG(2) << "fail to create class: "
                        << config["file_system"]["class"].as<std::string>();
                return -1;
            }
        } else {
            _file_system.reset(CREATE_CLASS(FileSystem, "LocalFileSystem"));
            if (_file_system == nullptr || _file_system->initialize(YAML::Load(""), context) != 0) {
                VLOG(2) << "fail to init file system";
                return -1;
            }
        }
        return 0;
    }

    //判断样本数据是否已就绪，就绪表明可以开始download
    virtual bool is_data_ready(const std::string& data_dir) {
        auto done_file_path = _file_system->path_join(data_dir, _done_file_name);
        if (_file_system->exists(done_file_path)) {
            return true;
        }
        return false;
    }

    virtual std::vector<std::string> data_file_list(const std::string& data_dir) {
        std::vector<std::string> data_files;
        for (auto& filepath : _file_system->list(data_dir)) {
            auto filename = _file_system->path_split(filepath).second;
            if (filename != _done_file_name &&
                string::begin_with(filename, _filename_prefix)) {
                data_files.push_back(std::move(filepath));
            }
        }
        return data_files;
    }

    //读取数据样本流中
    virtual int read_all(const std::string& data_dir, framework::Channel<DataItem> data_channel) {
        auto deleter = [](framework::ChannelWriter<DataItem> *writer) {
            if (writer) {
                writer->Flush();
                VLOG(3) << "writer auto flush";
            }
            delete writer;
        };
        std::unique_ptr<framework::ChannelWriter<DataItem>, decltype(deleter)> writer(new framework::ChannelWriter<DataItem>(data_channel.get()), deleter);
        DataItem data_item;

        auto file_list = data_file_list(data_dir);
        int file_list_size = file_list.size();

        VLOG(5) << "omg max_threads: " << omp_get_max_threads();
        #pragma omp parallel for
        for (int i = 0; i < file_list_size; ++i) {
            VLOG(5) << "omg num_threads: " << omp_get_num_threads() << ", start read: " << i << std::endl;
        }
        for (int i = 0; i < file_list_size; ++i) {
            //VLOG(5) << "omg num_threads: " << omp_get_num_threads() << ", start read: " << i;
            const auto& filepath = file_list[i];
            {
                std::shared_ptr<FILE> fin = _file_system->open_read(filepath, _pipeline_cmd);
                if (fin == nullptr) {
                    VLOG(2) << "fail to open file: " << filepath << ", with cmd: " << _pipeline_cmd;
                    return -1;
                }
                char *buffer = nullptr;
                size_t buffer_size = 0;
                ssize_t line_len = 0;
                while ((line_len = getline(&buffer, &buffer_size, fin.get())) != -1) {
                    if (line_len > 0 && buffer[line_len - 1] == '\n') {
                        buffer[--line_len] = '\0';
                    }
                    if (line_len <= 0) {
                        continue;
                    }
                    if (_parser->parse(buffer, line_len, data_item) == 0) {
                        (*writer) << std::move(data_item);
                    }
                }
                if (buffer != nullptr) {
                    free(buffer);
                    buffer = nullptr;
                    buffer_size = 0;
                }
                if (ferror(fin.get()) != 0) {
                    VLOG(2) << "fail to read file: " << filepath;
                    return -1;
                }
            }
            if (_file_system->err_no() != 0) {
                _file_system->reset_err_no();
                return -1;
            }
        }
        writer->Flush();
        if (!(*writer)) {
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
    std::string _done_file_name;  // without data_dirq
    std::string _filename_prefix;
    std::unique_ptr<FileSystem> _file_system;
};
REGISTER_CLASS(DataReader, LineDataReader);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
