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

int FileSystem::append_line(const std::string& path, 
    const std::string& line, size_t reserve_line_num) {
    std::string tail_data;
    if (exists(path)) {
        tail_data = paddle::string::trim_spaces(tail(path, reserve_line_num)); 
    }
    if (tail_data.size() > 0) {
        tail_data = tail_data + "\n" + line;
    } else {
        tail_data = line;
    }
    VLOG(2) << "Append to file:" << path << ", line str:" << line;
    while (true) {
        remove(path);
        auto fp = open_write(path, "");
        if (fwrite(tail_data.c_str(), tail_data.length(), 1, &*fp) == 1) {
            break;
        }     
        sleep(10);   
        VLOG(0) << "Retry Append to file:" << path << ", line str:" << line;
    }
    return 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
