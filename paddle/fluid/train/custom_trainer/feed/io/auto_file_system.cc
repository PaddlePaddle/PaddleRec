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

#include "paddle/fluid/train/custom_trainer/feed/io/shell.h"
#include "paddle/fluid/string/string_helper.h"
#include "glog/logging.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class AutoFileSystem : public FileSystem {
public:
    int initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context) override {
        _file_system.clear();
        if (config) {
            for (auto& prefix_fs: config) {
                std::unique_ptr<FileSystem> fs(CREATE_CLASS(FileSystem, prefix_fs.second["class"].as<std::string>("")));
                if (fs == nullptr) {
                    VLOG(2) << "fail to create class: " << prefix_fs.second["class"].as<std::string>("");
                    return -1;
                }
                if (fs->initialize(prefix_fs.second, context) != 0) {
                    VLOG(2) << "fail to initialize class: " << prefix_fs.second["class"].as<std::string>("");
                    return 0;
                }
                _file_system.emplace(prefix_fs.first.as<std::string>(""), std::move(fs));
            }
        }
        if (_file_system.find("default") == _file_system.end()) {
            std::unique_ptr<FileSystem> fs(CREATE_CLASS(FileSystem, "LocalFileSystem"));
            if (fs == nullptr || fs->initialize(YAML::Load(""), context) != 0) {
                return -1;
            }
            _file_system.emplace("default", std::move(fs));
        }
        return 0;
    }

    std::shared_ptr<FILE> open_read(const std::string& path, const std::string& converter)
            override {
        return get_file_system(path)->open_read(path, converter);
    }

    std::shared_ptr<FILE> open_write(const std::string& path, const std::string& converter)
            override {
        return get_file_system(path)->open_write(path, converter);
    }

    int64_t file_size(const std::string& path) override {
        return get_file_system(path)->file_size(path);
    }

    void remove(const std::string& path) override {
        get_file_system(path)->remove(path);
    }

    std::vector<std::string> list(const std::string& path) override {
        return get_file_system(path)->list(path);
    }

    std::string tail(const std::string& path) override {
        return get_file_system(path)->tail(path);
    }

    bool exists(const std::string& path) override {
        return get_file_system(path)->exists(path);
    }

    void mkdir(const std::string& path) override {
        get_file_system(path)->mkdir(path);
    }

    FileSystem* get_file_system(const std::string& path) {
        auto pos = path.find_first_of(":");
        if (pos != std::string::npos) {
            auto substr = path.substr(0, pos + 1);
            auto fs_it = _file_system.find(substr);
            if (fs_it != _file_system.end()) {
                return fs_it->second.get();
            }
        }
        VLOG(5) << "path: " << path << ", select default file system";
        return _file_system["default"].get();
    }

    int err_no() const override {
        if (_err_no == 0) {
            for (const auto& file_system : _file_system) {
                if (file_system.second->err_no() != 0) {
                    const_cast<int&>(_err_no) = -1;
                    break;
                }
            }
        }
        return FileSystem::err_no();
    }

    void reset_err_no() override {
        _err_no = 0;
        for (auto& file_system : _file_system) {
            file_system.second->reset_err_no();
        }
    }

private:
    std::unordered_map<std::string, std::unique_ptr<FileSystem>> _file_system;
};
REGISTER_CLASS(FileSystem, AutoFileSystem);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
