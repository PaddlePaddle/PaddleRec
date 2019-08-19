#include "paddle/fluid/train/custom_trainer/feed/dataset/data_reader.h"

#include <cstdio>
#include <atomic>

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

/********************************
 * feasign压缩格式
 * 情形1：slot:hot
 * |4b|4b|4b|4b|4b| 28b |
 * |slot       |0 |sign |
 * 情形2：slot:hot*n
 * |4b|4b|4b|4b|4b|4b|4b|4b|32b*n|
 * |slot       |1 |0 |len  |sign |
 * 情形3：slot:cold
 * |4b|4b|4b|4b|4b|4b| 64b |
 * |slot       |2 |0 |sign |
 * 情形4：slot:cold*n
 * |4b|4b|4b|4b|4b|4b|4b|4b|64b*n|
 * |slot       |3 |0 |len  |sign |
 ********************************/
class ArchiveDataParse : public DataParser {
public:
    static const uint8_t HOT_SIGN_SIZE = 4;
    static const uint8_t COLD_SIGN_SIZE = 8;

public:
    ArchiveDataParse() {}
    virtual ~ArchiveDataParse() {}

    struct Record {
        int show, clk;
        std::string tags;
        std::map<std::string, std::vector<float>> vec_feas;
        int sample_type;
        std::map<std::string, std::vector<int>> auc_category_info_map; //为细维度计算auc准备的数据
        std::vector<FeatureItem> hot_feas, cold_feas; //冷(int32_t)热(uint64_t)feasign

        void clear() {
            show = 0;
            clk = 0;
            tags.clear();
            vec_feas.clear();
            sample_type = 0;
            auc_category_info_map.clear();
            hot_feas.clear();
            cold_feas.clear();
        }

        uint32_t calc_compress_feas_lens() const {
            uint32_t hot_len = hot_feas.size();
            uint32_t cold_len = cold_feas.size();
            uint32_t cursor = 0;
            int32_t pre_slot = -1;
            uint32_t k = 0;
            //热编码
            if (hot_len > 0) {
                pre_slot = hot_feas[0].slot();

                for (uint32_t i = 0; i < hot_len + 1; ++i) {
                    if (i == hot_len || pre_slot != hot_feas[i].slot()) {
                        cursor += 2;
                        //情形2
                        if (i - k > 1) {
                            cursor += 2;
                        }
                        //情形1/2
                        cursor += (HOT_SIGN_SIZE * (i - k));
                        k = i;
                    }
                    pre_slot = hot_feas[i].slot();
                }
            }
            //冷编码
            if (cold_len > 0) {
                pre_slot = cold_feas[0].slot();
                k = 0;

                for (uint32_t i = 0; i < cold_len + 1; ++i) {
                    if (i == cold_len || pre_slot != cold_feas[i].slot()) {
                        cursor += 2;
                        //情形4
                        if (i - k > 1) {
                            cursor += 2;
                        } else { //情形3
                            cursor++;
                        }
                        //情形3/4
                        cursor += (COLD_SIGN_SIZE * (i - k));
                        k = i;
                    }
                    pre_slot = cold_feas[i].slot();
                }
            }
            return cursor;
        }

        void parse_feas(char* buffer) const {
            if (buffer == nullptr) {
                return ;
            }
            uint32_t cursor = 0;
            uint32_t hot_len = hot_feas.size();
            uint32_t cold_len = cold_feas.size();
            int32_t pre_slot = -1;
            int32_t hot_sign;
            uint16_t slot;
            uint8_t flag = 0, len = 0;
            uint32_t k = 0;
            //热编码
            if (hot_len > 0) {
                pre_slot = hot_feas[0].slot();

                for (uint32_t i = 0; i < hot_len + 1; ++i) {
                    if (i == hot_len || pre_slot != hot_feas[i].slot()) {
                        memcpy(buffer + cursor, &pre_slot, 2);
                        cursor += 2;
                        //情形2
                        if (i - k > 1) {
                            flag = 0x10;
                            memcpy(buffer + cursor, &flag, 1);
                            cursor++;
                            len = i - k;
                            memcpy(buffer + cursor, &len, 1);
                            cursor++;
                        }
                        //情形1/2
                        for (uint32_t j = k; j < i; ++j) {
                            hot_sign = (int32_t) hot_feas[j].sign();
                            for (uint8_t b = 0; b < HOT_SIGN_SIZE; ++b) {
                                flag = (hot_sign >> ((HOT_SIGN_SIZE - b - 1) * 8)) & 0xFF;
                                memcpy(buffer + cursor, &flag, 1);
                                cursor++;
                            }
                        }
                        k = i;
                    }
                    pre_slot = hot_feas[i].slot();
                }
            }
            //冷编码
            if (cold_len > 0) {
                pre_slot = cold_feas[0].slot();
                k = 0;

                for (uint32_t i = 0; i < cold_len + 1; ++i) {
                    if (i == cold_len || pre_slot != cold_feas[i].slot()) {
                        memcpy(buffer + cursor, &pre_slot, 2);
                        cursor += 2;
                        //情形4
                        if (i - k > 1) {
                            flag = 0x30;
                            memcpy(buffer + cursor, &flag, 1);
                            cursor++;
                            len = i - k;
                            memcpy(buffer + cursor, &len, 1);
                            cursor++;
                        }
                        //情形3/4
                        for (uint32_t j = k; j < i; ++j) {
                            if (i - k == 1) {
                                flag = 0x20;
                                memcpy(buffer + cursor, &flag, 1);
                                cursor++;
                            }
                            memcpy(buffer + cursor, &cold_feas[j].sign(), COLD_SIGN_SIZE);
                            cursor += COLD_SIGN_SIZE;
                        }
                        k = i;
                    }
                    pre_slot = cold_feas[i].slot();
                }
            }
        }
    };

    virtual int initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context) {
        _index = context->cache_dict;

        return 0;
    }

    virtual int parse(const char* str, size_t len, DataItem& data) const {
        size_t pos = paddle::string::count_nonspaces(str);
        if (pos >= len) {
            VLOG(2) << "fail to parse line: " << std::string(str, len) << ", strlen: " << len;
            return -1;
        }
        VLOG(5) << "getline: " << str << " , pos: " << pos << ", len: " << len;
        data.id.assign(str, pos);
        str += pos;

        static thread_local std::vector<float> vec_feas;
        static thread_local Record rec;
        rec.clear();

        const char* line_end = str + len;
        char* cursor = NULL;
        CHECK((rec.show = (int)strtol(str, &cursor, 10), cursor != str));
        str = cursor;
        CHECK((rec.clk = (int)strtol(str, &cursor, 10), cursor != str));
        str = cursor;
        CHECK(rec.show >= 1 && rec.clk >= 0 && rec.clk <= rec.show);

        while (*(str += paddle::string::count_nonspaces(str)) != 0) {
            if (*str == '*') {
                str++;
                size_t len = paddle::string::count_nonspaces(str);
                std::string tag(str, str + len);
                rec.tags = tag;
                str += len;
            } else if (*str == '$') {
                str++;
                CHECK((rec.sample_type = (int)strtol(str, &cursor, 10), cursor != str))<<" sample type parse err:" << str;
                str = cursor;
            } else if (*str == '#') {
                str++;
                size_t len = std::find_if_not(str, line_end,
                                              [](char c) { return std::isalnum(c) != 0 || c == '_';}) - str;
                CHECK(len > 0 && *(str + len) == ':');
                std::string name(str, len);
                str += len;
                vec_feas.clear();
                while (*str == ':') {
                    float val = 0;
                    CHECK((val = strtof(str + 1, &cursor), cursor > str));
                    vec_feas.push_back(val);
                    str = cursor;
                }
                CHECK(rec.vec_feas.insert({name, vec_feas}).second);
            } else if (*str == '@') {
                str++;
                size_t len = paddle::string::count_nonspaces(str);
                std::string all_str(str, str + len);
                str += len;
                //category_name1=value1,value2,value3|category_name2=value1,value2|....
                std::vector<std::string> all_category_vec = paddle::string::split_string(all_str, "|");
                for (size_t i = 0; i < all_category_vec.size(); ++i) {
                    std::string& single_category_str = all_category_vec[i];
                    std::vector<std::string> str_vec = paddle::string::split_string(single_category_str, "=");
                    CHECK(str_vec.size() == 2);
                    /*std::string category_name = str_vec[0];
                    std::vector<int> category_info_vec = paddle::string::split_string<int>(str_vec[1], ",");
                    CHECK(category_info_vec.size() > 0);

                    CHECK(rec.auc_category_info_map.insert({category_name, category_info_vec}).second);*/
                }
            } else {
                uint64_t sign = 0;
                int slot = -1;
                CHECK((sign = (uint64_t) strtoull(str, &cursor, 10), cursor != str));
                str = cursor;
                CHECK(*str++ == ':');
                CHECK(!isspace(*str));
                CHECK((slot = (int) strtol(str, &cursor, 10), cursor != str)) << " format error: " << str;
                CHECK((uint16_t) slot == slot);
                str = cursor;

                int32_t compress_sign = _index->sign2index(sign);
                if (compress_sign < 0) {
                    rec.cold_feas.emplace_back(sign, (uint16_t)slot);
                } else {
                    rec.hot_feas.emplace_back(compress_sign, (uint16_t)slot);
                }
            }
        }

        paddle::framework::BinaryArchive bar;
        bar << rec.show;
        bar << rec.clk;
        bar << rec.tags;
        bar << rec.vec_feas;
        bar << rec.sample_type;
        bar << rec.auc_category_info_map;
        uint32_t feas_len = rec.calc_compress_feas_lens();
        bar << feas_len;
        bar.Resize(bar.Length() + feas_len);
        rec.parse_feas(bar.Cursor());
        data.data.assign(bar.Buffer(), bar.Length());

        return 0;
    }

    virtual int parse(const char* str, DataItem& data) const {

    }

    virtual int parse_to_sample(const DataItem& data, SampleInstance& instance) const {
        instance.id = data.id;
        if (data.data.empty()) {
            return -1;
        }

        //FIXME temp
        int show = 0, clk = 0;
        std::string tags;
        std::map<std::string, std::vector<float>> vec_feas;
        int sample_type = 0;
        std::map<std::string, std::vector<int>> auc_category_info_map;
        uint32_t feas_len = 0;

        paddle::framework::BinaryArchive bar;
        bar.SetReadBuffer(const_cast<char*>(&data.data[0]), data.data.size(), nullptr);

        bar >> show;
        bar >> clk;
        bar >> tags;
        bar >> vec_feas;
        bar >> sample_type;
        bar >> auc_category_info_map;
        bar >> feas_len;
        parse_feas_to_ins(bar.Cursor(), feas_len, instance.features);

        return 0;
    }

private:
    void parse_feas_to_ins(char* buffer, uint32_t len, std::vector<FeatureItem>& ins) const {
        if (buffer == nullptr) {
            return ;
        }

        uint32_t cursor = 0;
        uint16_t slot;
        uint8_t flag;
        while (cursor < len) {
            memcpy(&slot, buffer + cursor, 2);
            cursor += 2;

            memcpy(&flag, buffer + cursor, 1);
            flag &= 0xF0;

            CHECK(flag == 0x00 || flag == 0x10|| flag == 0x20 || flag == 0x30);

            if (flag == 0x00 || flag == 0x10) {
                uint8_t len = 1;
                if (flag == 0x10) {
                    cursor++;
                    memcpy(&len, buffer + cursor, 1);
                    cursor++;
                }
                for (uint8_t i = 0; i < len; ++i) {
                    int32_t sign;
                    for (uint8_t j = 0; j < HOT_SIGN_SIZE; ++j) {
                        memcpy((char*)&sign + HOT_SIGN_SIZE-j-1, buffer + cursor, 1);
                        cursor++;
                    }

                    uint64_t sign64 = sign & 0x0FFFFFFF;
                    sign64 = _index->index2sign((int32_t)sign64);
                    ins.emplace_back(sign64, slot);
                }
            }

            if (flag == 0x20 || flag == 0x30) {
                uint8_t len = 1;
                cursor++;
                if (flag == 0x30) {
                    memcpy(&len, buffer + cursor, 1);
                    cursor++;
                }

                for (uint8_t i = 0; i < len; ++i) {
                    uint64_t sign64;
                    memcpy(&sign64, buffer + cursor, COLD_SIGN_SIZE);
                    cursor += COLD_SIGN_SIZE;
                    ins.emplace_back(sign64, slot);
                }
            }
        }
    }

private:
    std::shared_ptr<SignCacheDict> _index;

};
REGISTER_CLASS(DataParser, ArchiveDataParse);

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
        } else if (context->file_system != nullptr) { 
            _file_system = context->file_system;
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
        auto file_list = data_file_list(data_dir);
        return read_all(file_list, data_channel);
    }
    virtual int read_all(const std::vector<std::string>& file_list, ::paddle::framework::Channel<DataItem> data_channel) {
        const int file_list_size = file_list.size();
        std::atomic<bool> is_failed(false);

        const int max_threads = omp_get_max_threads();
        std::vector<framework::ChannelWriter<DataItem>> writers; // writer is not thread safe
        writers.reserve(max_threads);
        for (int i = 0; i < max_threads; ++i) {
            writers.emplace_back(data_channel.get());
        }
        VLOG(5) << "file_list: " << string::join_strings(file_list, ' ');
        #pragma omp parallel for
        for (int i = 0; i < file_list_size; ++i) {
            if (is_failed) {
                continue;
            }
            const int thread_num = omp_get_thread_num();
            framework::ChannelWriter<DataItem> *writer = nullptr;
            if (thread_num < max_threads) {
                writer = &writers[thread_num];
            }
            const auto& filepath = file_list[i];
            std::shared_ptr<FILE> fin = _file_system->open_read(filepath, _pipeline_cmd);
            if (fin == nullptr) {
                VLOG(2) << "fail to open file: " << filepath << ", with cmd: " << _pipeline_cmd;
                is_failed = true;
                continue;
            }
            char *buffer = nullptr;
            size_t buffer_size = 0;
            ssize_t line_len = 0;
            while ((line_len = getline(&buffer, &buffer_size, fin.get())) != -1) {
                // 去掉行位回车
                if (line_len > 0 && buffer[line_len - 1] == '\n') {
                    buffer[--line_len] = '\0';
                }
                // 忽略空行
                if (line_len <= 0) {
                    continue;
                }
                DataItem data_item;
                if (_parser->parse(buffer, line_len, data_item) == 0) {
                    VLOG(5) << "parse data: " << data_item.id << " " << data_item.data << ", filename: " << filepath << ", thread_num: " << thread_num << ", max_threads: " << max_threads;
                    if (writer == nullptr) {
                        if (!data_channel->Put(std::move(data_item))) {
                            VLOG(2) << "fail to put data, thread_num: " << thread_num;
                        }
                    } else {
                        (*writer) << std::move(data_item);
                    }
                }
            }
            if (buffer != nullptr) {
                free(buffer);
                buffer = nullptr;
                buffer_size = 0;
            }
            if (ferror(fin.get()) != 0) {
                VLOG(2) << "fail to read file: " << filepath;
                is_failed = true;
                continue;
            }
            if (_file_system->err_no() != 0) {
                _file_system->reset_err_no();
                is_failed = true;
                continue;
            }
        }
        // omp end

        for (int i = 0; i < max_threads; ++i) {
            writers[i].Flush();
            if (!writers[i]) {
                VLOG(2) << "writer " << i << " is failed";
                is_failed = true;
            }
        }
        data_channel->Close();
        return is_failed ? -1 : 0;
    }

    virtual const DataParser* get_parser() {
        return _parser.get();
    }

private:
    std::string _done_file_name;  // without data_dir
    std::string _filename_prefix;
    std::shared_ptr<FileSystem> _file_system;
};
REGISTER_CLASS(DataReader, LineDataReader);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
