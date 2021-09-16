// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef BCLOUD
#include "baidu/rpc/channel.h"
#include "baidu/rpc/parallel_channel.h"
#include "rapidjson_1.0/document.h"
#include "rapidjson_1.0/rapidjson.h"
#else
#include "brpc/channel.h"
#include "brpc/parallel_channel.h"
#include "butil/third_party/rapidjson/document.h"
#endif
#include "bvar/bvar.h"

#ifdef BCLOUD
namespace brpc = baidu::rpc;
#ifndef BUTIL_RAPIDJSON_NAMESPACE
#define BUTIL_RAPIDJSON_NAMESPACE RAPIDJSON_NAMESPACE
#endif
#endif

namespace rec {
namespace mcube {

struct MetaInfo {
  MetaInfo() : dict_name(""), shard_num(0), dup_num(0) {}
  ~MetaInfo() {
    for (size_t i = 0; i != cube_conn.size(); ++i) {
      if (cube_conn[i] != NULL) {
        delete cube_conn[i];
        cube_conn[i] = NULL;
      }
    }
    cube_conn.clear();
  }

  std::string dict_name;
  int shard_num;
  int dup_num;
  std::vector<::brpc::Channel*> cube_conn;
  bvar::Adder<uint64_t> cube_request_num;
  bvar::Adder<uint64_t> cube_rpcfail_num;
};

class Meta {
 public:
  static Meta* instance();

 public:
  ~Meta();

  int init(const char* conf_file);

  int destroy();

  MetaInfo* get_meta(const std::string& dict_name);

  const std::vector<const MetaInfo*> metas();

  void reset_bvar(const std::string& dict_name);

 private:
  MetaInfo* create_meta(const BUTIL_RAPIDJSON_NAMESPACE::Value& meta_config);

  int create_meta_from_ipport(MetaInfo* meta,
                              const BUTIL_RAPIDJSON_NAMESPACE::Value& nodes,
                              int timeout,
                              int retry,
                              int backup_request);

  int create_meta_from_ipport_list(
      MetaInfo* meta,
      const BUTIL_RAPIDJSON_NAMESPACE::Value& nodes,
      int timeout,
      int retry,
      int backup_request,
      const std::string& load_balancer);

  int create_meta_from_ipport_parallel(
      MetaInfo* meta,
      const BUTIL_RAPIDJSON_NAMESPACE::Value& nodes,
      int timeout,
      int retry,
      int backup_request);

 private:
  Meta() {}
  Meta(const Meta&) {}

 private:
  std::unordered_map<std::string, MetaInfo*> _metas;
};  // class Meta

}  // namespace mcube
}  // namespace rec
