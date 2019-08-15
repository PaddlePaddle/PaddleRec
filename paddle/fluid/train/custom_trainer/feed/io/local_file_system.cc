#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"

#include <string>

#include "paddle/fluid/train/custom_trainer/feed/io/shell.h"
#include "paddle/fluid/string/string_helper.h"
#include "glog/logging.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class LocalFileSystem : public FileSystem {
public:
    int initialize(const YAML::Node& config, std::shared_ptr<TrainerContext> context) override {
        _buffer_size = config["buffer_size"].as<size_t>(0);
        return 0;
    }

    std::shared_ptr<FILE> open_read(const std::string& path, const std::string& converter) override {
        std::string cmd = path;
        bool is_pipe = false;
        if (string::end_with(path, ".gz")) {
            shell_add_read_converter(cmd, is_pipe, "zcat");
        }

        shell_add_read_converter(cmd, is_pipe, converter);
        return shell_open(cmd, is_pipe, "r", _buffer_size);
    }

    std::shared_ptr<FILE> open_write(const std::string& path, const std::string& converter) override {
        std::string cmd = path;

        shell_execute(string::format_string("mkdir -p $(dirname \"%s\")", path.c_str()));

        bool is_pipe = false;

        if (string::end_with(path, ".gz")) {
            shell_add_write_converter(cmd, is_pipe, "gzip");
        }

        shell_add_write_converter(cmd, is_pipe, converter);
        return shell_open(cmd, is_pipe, "w", _buffer_size);
    }

    int64_t file_size(const std::string& path) override {
        struct stat buf;
        if (0 != stat(path.c_str(), &buf)) {
            LOG(FATAL) << "file stat not zero";
            return -1;
        }
        return (int64_t)buf.st_size;
    }

    void remove(const std::string& path) override {
        if (path == "") {
            return;
        }

        shell_execute(string::format_string("rm -rf %s", path.c_str()));
    }

    std::vector<std::string> list(const std::string& path) override {
        if (path == "") {
            return {};
        }

        std::shared_ptr<FILE> pipe;
        pipe = shell_popen(
                string::format_string("find %s -maxdepth 1 -type f", path.c_str()), "r", &_err_no);
        string::LineFileReader reader;
        std::vector<std::string> list;

        while (reader.getline(&*pipe)) {
            list.push_back(reader.get());
        }

        return list;
    }

    std::string tail(const std::string& path) override {
        if (path == "") {
            return "";
        }

        return shell_get_command_output(string::format_string("tail -1 %s ", path.c_str()));
    }

    bool exists(const std::string& path) override {
        std::string test_f = shell_get_command_output(
                string::format_string("[ -f %s ] ; echo $?", path.c_str()));

        if (string::trim_spaces(test_f) == "0") {
            return true;
        }

        std::string test_d = shell_get_command_output(
                string::format_string("[ -d %s ] ; echo $?", path.c_str()));

        if (string::trim_spaces(test_d) == "0") {
            return true;
        }

        return false;
    }

    void mkdir(const std::string& path) override {
        if (path == "") {
            return;
        }

        shell_execute(string::format_string("mkdir -p %s", path.c_str()));
    }

private:
    size_t _buffer_size = 0;
};
REGISTER_CLASS(FileSystem, LocalFileSystem);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
