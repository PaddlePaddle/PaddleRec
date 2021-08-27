// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils.h"

namespace string {
template <class... ARGS>
std::string format_string(const char* fmt, ARGS&&... args) {
  std::string str;
  format_string_append(str, fmt, args...);
  return std::move(str);
}

template <class... ARGS>
std::string format_string(const std::string& fmt, ARGS&&... args) {
  return format_string(fmt.c_str(), args...);
}   
}
class DataFeed {
public:
    DataFeed() {};
    size_t hdfs_buffer_size()
    {
        static size_t x = 0;
        return x;
    }

    const std::string& hdfs_command()
    {
        static std::string x = "hadoop fs";
        return x;
    }

    const std::string& download_cmd()
    {
        static std::string x = "";
        return x;
    }

    bool fs_begin_with_internal(const std::string& path,
                                const std::string& str)
    {
        return strncmp(path.c_str(), str.c_str(), str.length()) == 0;
    }

    int fs_select_internal(const std::string& path) 
    {
        if (fs_begin_with_internal(path, "hdfs:")) {
            return 1;
        } else if (fs_begin_with_internal(path, "afs:")) {
            return 1;
        }
        return 0;
    }

    std::shared_ptr<FILE> localfs_open_read(std::string path,
                                            const std::string& converter)
    {
        std::shared_ptr<FILE> fp = nullptr;
        return fp;
    }
    
    static bool fs_end_with_internal(const std::string& path,
                                 const std::string& str)
    {
        return path.length() >= str.length() &&
            strncmp(&path[path.length() - str.length()], str.c_str(),
                    str.length()) == 0;
    }

    static void fs_add_read_converter_internal(std::string& path,  // NOLINT
                                           bool& is_pipe,      // NOLINT
                                           const std::string& converter)
    {
        if (converter == "") {
            return;
        }

        if (!is_pipe) {
            path = string::format_string("( %s ) < \"%s\"", converter.c_str(),
                                        path.c_str());
            is_pipe = true;
        } else {
            path = string::format_string("%s | %s", path.c_str(), converter.c_str());
        }
    }

    static std::shared_ptr<FILE> fs_open_internal(const std::string& path,
                                              bool is_pipe,
                                              const std::string& mode,
                                              size_t buffer_size,
                                              int* err_no = 0) 
    {
        std::shared_ptr<FILE> fp = nullptr;
        /*
        if (!is_pipe) {
            fp = shell_fopen(path, mode);
        } else {
            fp = shell_popen(path, mode, err_no);
        }

        if (buffer_size > 0) {
            char* buffer = new char[buffer_size];
            CHECK_EQ(0, setvbuf(&*fp, buffer, _IOFBF, buffer_size));
            fp = {&*fp, [fp, buffer](FILE*) mutable {  // NOLINT
                    CHECK(fp.unique());                // NOLINT
                    fp = nullptr;
                    delete[] buffer;
                }};
        }
        */
        return fp;
    }

    std::shared_ptr<FILE> hdfs_open_read(std::string path, int* err_no,
                                         const std::string& converter)
    {
        if (fs_end_with_internal(path, ".gz")) {
            path = string::format_string("%s -text \"%s\"", hdfs_command().c_str(),
                                 path.c_str());
        } else {
            const std::string file_path = path;
            path = string::format_string("%s -cat \"%s\"", hdfs_command().c_str(),
                                        file_path.c_str());
            if (download_cmd() != "") {  // use customized download command
            path = string::format_string("%s \"%s\"", download_cmd().c_str(),
                                        file_path.c_str());
            }
        }
        bool is_pipe = true;
        fs_add_read_converter_internal(path, is_pipe, converter);
        return fs_open_internal(path, is_pipe, "r", hdfs_buffer_size(), err_no);
    }

    std::shared_ptr<FILE> fs_open_read(const std::string& path, int* err_no, 
                                       const std::string& converter)
    {
        switch (fs_select_internal(path)) {
            case 0:
                return localfs_open_read(path, converter);
            case 1:
                return hdfs_open_read(path, err_no, converter);
            default:
                LOG(ERROR) << "Unsupport file system. Now only support local file system and hdfs.";
        }
    }
};
