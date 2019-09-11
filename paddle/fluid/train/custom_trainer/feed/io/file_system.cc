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

int FileSystem::copy(const std::string& ori_path, const std::string& dest_path) {
    if (!exists(ori_path)) {
        return -1;
    }
    remove(dest_path);
    auto ori_file = open_read(ori_path, "");
    auto dest_file = open_write(dest_path, "");
    size_t read_buffer_size = 102400; // 100kb
    char* buffer = new char[read_buffer_size];
    while (true) {
        size_t read_size = fread(buffer, 1, read_buffer_size, ori_file.get());
        CHECK(ferror(ori_file.get()) == 0) << " File read Failed:" << ori_path;
        if (read_size > 0) {
            fwrite(buffer, 1, read_size, dest_file.get());
        } 
        // read done
        if (read_size < read_buffer_size) {
            break;
        }
    }
    delete[] buffer; 
    return 0;
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
        {
            auto fp = open_write(path, "");
            if (fwrite(tail_data.c_str(), tail_data.length(), 1, &*fp) == 1) {
                break;
            }
        }     
        sleep(10);   
        VLOG(0) << "Retry Append to file:" << path << ", line str:" << line;
    }
    return 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
