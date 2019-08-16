/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"

#include <string>
#include <unordered_map>
#include <tuple>

#include "paddle/fluid/train/custom_trainer/feed/io/shell.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/fluid/string/piece.h"
#include "glog/logging.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class HadoopFileSystem : public FileSystem {
public:
    int initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context) override {
        _buffer_size = config["buffer_size"].as<size_t>(0);
        _hdfs_command = config["hdfs_command"].as<std::string>("hadoop fs");
        _ugi.clear();
        if (config["ugis"] && config["ugis"].Type() == YAML::NodeType::Map) {
            for (const auto& prefix_ugi : config["ugis"]) {
                _ugi.emplace(prefix_ugi.first.as<std::string>(), prefix_ugi.second.as<std::string>());
            }
        }
        if (_ugi.find("default") == _ugi.end()) {
            VLOG(2) << "fail to load default ugi";
            return -1;
        }
        return 0;
    }

    std::shared_ptr<FILE> open_read(const std::string& path, const std::string& converter)
            override {
        std::string cmd;
        if (string::end_with(path, ".gz")) {
            cmd = string::format_string(
                    "%s -text \"%s\"", hdfs_command(path).c_str(), path.c_str());
        } else {
            cmd = string::format_string("%s -cat \"%s\"", hdfs_command(path).c_str(), path.c_str());
        }

        bool is_pipe = true;
        shell_add_read_converter(cmd, is_pipe, converter);
        return shell_open(cmd, is_pipe, "r", _buffer_size, &_err_no);
    }

    std::shared_ptr<FILE> open_write(const std::string& path, const std::string& converter)
            override {
        std::string cmd =
                string::format_string("%s -put - \"%s\"", hdfs_command(path).c_str(), path.c_str());
        bool is_pipe = true;

        if (string::end_with(path, ".gz\"")) {
            shell_add_write_converter(cmd, is_pipe, "gzip");
        }

        shell_add_write_converter(cmd, is_pipe, converter);
        return shell_open(cmd, is_pipe, "w", _buffer_size, &_err_no);
    }

    int64_t file_size(const std::string& path) override {
        _err_no = -1;
        VLOG(2) << "not support";
        return 0;
    }

    void remove(const std::string& path) override {
        if (path == "") {
            return;
        }

        shell_execute(string::format_string(
                "%s -rmr %s &>/dev/null; true", _hdfs_command.c_str(), path.c_str()));
    }

    std::vector<std::string> list(const std::string& path) override {
        if (path == "") {
            return {};
        }
        auto paths = split_path(path);

        int err_no = 0;
        std::vector<std::string> list;
        do {
            err_no = 0;
            std::shared_ptr<FILE> pipe;
            pipe = shell_popen(
                    string::format_string(
                            "%s -ls %s | ( grep ^- ; [ $? != 2 ] )",
                            hdfs_command(path).c_str(),
                            path.c_str()),
                    "r",
                    &err_no);
            string::LineFileReader reader;
            list.clear();

            while (reader.getline(&*pipe)) {
                std::vector<std::string> line = string::split_string(reader.get());
                if (line.size() != 8) {
                    continue;
                }
                list.push_back(get_prefix(paths) + line[7]);
            }
        } while (err_no == -1);
        return list;
    }

    std::string tail(const std::string& path) override {
        if (path == "") {
            return "";
        }

        return shell_get_command_output(string::format_string(
                "%s -text %s | tail -1 ", hdfs_command(path).c_str(), path.c_str()));
    }

    bool exists(const std::string& path) override {
        std::string test = shell_get_command_output(string::format_string(
                "%s -test -e %s ; echo $?", hdfs_command(path).c_str(), path.c_str()));

        if (string::trim_spaces(test) == "0") {
            return true;
        }

        return false;
    }

    void mkdir(const std::string& path) override {
        if (path == "") {
            return;
        }

        shell_execute(string::format_string(
                "%s -mkdir %s; true", hdfs_command(path).c_str(), path.c_str()));
    }

    std::string hdfs_command(const std::string& path) {
        auto paths = split_path(path);
        auto it = _ugi.find(std::get<1>(paths).ToString());
        if (it != _ugi.end()) {
            return hdfs_command_with_ugi(it->second);
        }
        VLOG(5) << "path: " << path << ", select default ugi";
        return hdfs_command_with_ugi(_ugi["default"]);
    }

    std::string hdfs_command_with_ugi(std::string ugi) {
        return string::format_string(
                "%s -Dhadoop.job.ugi=\"%s\"", _hdfs_command.c_str(), ugi.c_str());
    }

private:
    std::string get_prefix(const std::tuple<string::Piece, string::Piece, string::Piece>& paths) {
        if (std::get<1>(paths).len() == 0) {
            return std::get<0>(paths).ToString();
        }
        return std::get<0>(paths).ToString() + "//" + std::get<1>(paths).ToString();
    }

    // parse "xxx://abc.def:8756/user" as "xxx:", "abc.def:8756", "/user"
    // parse "xxx:/user" as "xxx:", "", "/user"
    // parse "xxx://abc.def:8756" as "xxx:", "abc.def:8756", ""
    // parse "other" as "", "", "other"
    std::tuple<string::Piece, string::Piece, string::Piece> split_path(string::Piece path) {
        std::tuple<string::Piece, string::Piece, string::Piece> result{string::SubStr(path, 0, 0), string::SubStr(path, 0, 0), path};
        auto fs_pos = string::Find(path, ':', 0) + 1;
        if (path.len() > fs_pos) {
            std::get<0>(result) = string::SubStr(path, 0, fs_pos);
            path = string::SkipPrefix(path, fs_pos);
            if (string::HasPrefix(path, "//")) {
                path = string::SkipPrefix(path, 2);
                auto end_pos = string::Find(path, '/', 0);
                if (end_pos != string::Piece::npos) {
                    std::get<1>(result) = string::SubStr(path, 0, end_pos);
                    std::get<2>(result) = string::SkipPrefix(path, end_pos);
                } else {
                    std::get<1>(result) = path;
                }
            } else {
                std::get<2>(result) = path;
            }
        }
        return result;
    }

    size_t _buffer_size = 0;
    std::string _hdfs_command;
    std::unordered_map<std::string, std::string> _ugi;
};
REGIST_CLASS(FileSystem, HadoopFileSystem);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
