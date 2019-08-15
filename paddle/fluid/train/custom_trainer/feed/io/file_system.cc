#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"
#include <string>

namespace paddle {
namespace custom_trainer {
namespace feed {

std::string FileSystem::path_join(const std::string& dir, const std::string& path) {
    if (dir.empty()) {
        return path;
    }
    if (dir.back() == '/') {
        return dir + path;
    }
    return dir + '/' + path;
}

std::pair<std::string, std::string> FileSystem::path_split(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return {".", path};
    }
    return {path.substr(0, pos), path.substr(pos + 1)};
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
