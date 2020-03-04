#include <iostream>
#include "paddle/fluid/feed/src/common/dict_plugin.h"
#include "paddle/fluid/framework/io/fs.h"

namespace paddle {
namespace framework {

int FeasignCacheDict::Load(
    const std::string& path, const std::string& converter) {
    auto version = version_ + 1;
    if (version >= versioned_entity_.size()) {
        version = 0;
    }
    auto& entity = versioned_entity_[version];
    uint64_t data_count = 0;
    auto file_list = fs_list(path);
    for (auto& file_path : file_list) {
        int err_no = 0;
        int line_len = 0;
        size_t buffer_size = 0;
        char *buffer = nullptr;
        char* data_ptr = NULL;
        auto file = fs_open_read(file_path, &err_no, converter);
        CHECK(err_no == 0);
        while ((line_len = getline(&buffer, &buffer_size, file.get())) > 0) {
            if (line_len <= 1) {
                continue;
            }
            ++data_count;
            entity.Append(strtoul(buffer, &data_ptr, 10), entity.Size());
        }
        if (buffer != nullptr) {
            free(buffer);
        }        
    }
    version_ = version;
    std::cerr << "Load success data_count" << data_count << " to version:" << version_ << std::endl;
    return 0;
}

}  // namespace framework
}  // namespace paddle
