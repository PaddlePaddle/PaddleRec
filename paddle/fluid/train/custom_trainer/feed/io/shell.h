#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/syscall.h>
#endif
#include <sys/types.h>
#ifndef _WIN32
#include <sys/wait.h>
#endif
#include <memory>
#include <string>
#include <utility>
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

inline bool& shell_verbose_internal() {
    static bool x = false;
    return x;
}

inline bool shell_verbose() {
    return shell_verbose_internal();
}

inline void shell_set_verbose(bool x) {
    shell_verbose_internal() = x;
}

extern std::shared_ptr<FILE> shell_fopen(const std::string& path, const std::string& mode);

extern std::shared_ptr<FILE> shell_popen(
        const std::string& cmd,
        const std::string& mode,
        int* err_no);

extern std::pair<std::shared_ptr<FILE>, std::shared_ptr<FILE>> shell_p2open(const std::string& cmd);

inline void shell_execute(const std::string& cmd) {
    int err_no = 0;
    do {
        err_no = 0;
        shell_popen(cmd, "w", &err_no);
    } while (err_no == -1);
}

extern std::string shell_get_command_output(const std::string& cmd);

extern void shell_add_read_converter(std::string& path, bool& is_pipe, const std::string& converter);

extern std::shared_ptr<FILE> shell_open(const std::string& path, bool is_pipe, const std::string& mode, size_t buffer_size, int* err_no = 0);

extern void shell_add_write_converter(std::string& path, bool& is_pipe, const std::string& converter);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
