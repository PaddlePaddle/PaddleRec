#pragma once

#include <memory>
#include <cstdio>
#include <vector>
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"
#include <yaml-cpp/yaml.h>

namespace paddle {
namespace custom_trainer {
namespace feed {
    
class FileSystem {
public:
    FileSystem() {}
    virtual ~FileSystem() {}
    virtual int initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context) = 0;
    virtual std::shared_ptr<FILE> open_read(const std::string& path, const std::string& converter) = 0;
    virtual std::shared_ptr<FILE> open_write(const std::string& path, const std::string& converter) = 0;
    virtual int64_t file_size(const std::string& path) = 0;
    virtual void remove(const std::string& path) = 0;
    virtual std::vector<std::string> list(const std::string& path) = 0;
    virtual std::string tail(const std::string& path, size_t tail_num = 1) = 0;
    virtual bool exists(const std::string& path) = 0;
    virtual void mkdir(const std::string& path) = 0;
    virtual std::string path_join(const std::string& dir, const std::string& path);
    template<class... STRS>
    std::string path_join(const std::string& dir, const std::string& path, const STRS&... paths) {
        return path_join(path_join(dir, path), paths...);
    }
    virtual std::pair<std::string, std::string> path_split(const std::string& path);
protected:
};
REGIST_REGISTERER(FileSystem);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
