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

#include "../include/meta.h"

#include <google/protobuf/descriptor.h>
#include <string.h>
#include <fstream>
#include <new>
#include <sstream>

#include "../proto/cube.pb.h"
#include "glog/logging.h"


namespace {
static ::rec::mcube::Meta* g_ins = NULL;
}

#ifdef BCLOUD
namespace brpc = baidu::rpc;
#endif

namespace rec {
namespace mcube {

Meta* Meta::instance() {
  if (g_ins == NULL) {
    g_ins = new Meta();
  }

  return g_ins;
}

Meta::~Meta() {}

int Meta::init(const char* conf_file) {
  std::ifstream ifs(conf_file);
  if (!ifs.is_open()) {
    LOG(ERROR) << "open conf file [" << conf_file << "]";
    return -1;
  }

  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());

  BUTIL_RAPIDJSON_NAMESPACE::Document document;
  document.Parse(content.c_str());

  int ret = 0;
  if (document.IsArray()) {
    for (auto it = document.Begin(); it != document.End(); ++it) {
      MetaInfo* info = create_meta(*it);

      if (info == NULL) {
        LOG(ERROR) << "create dict meta failed";
        ret = -1;
        break;
      }

      //LOG(INFO) << "init cube dict [" << info->dict_name << "] succ";
      _metas[info->dict_name] = info;
    }
  } else {
    LOG(ERROR)
        << "conf file [" << conf_file
        << "] json schema error, please ensure that the json is an array";
  }

  if (_metas.size() == 0) {
    LOG(ERROR) << "no valid cube";
    ret = -1;
  }

  return ret;
}

int Meta::destroy() {
  for (auto it = _metas.begin(); it != _metas.end(); ++it) {
    if (it->second != NULL) {
      delete it->second;
      it->second = NULL;
    }
  }
  _metas.clear();
  return 0;
}

MetaInfo* Meta::get_meta(const std::string& dict_name) {
  auto iter = _metas.find(dict_name);

  if (iter != _metas.end()) {
    return iter->second;
  }

  return NULL;
}

void Meta::reset_bvar(const std::string& dict_name) {
  auto iter = _metas.find(dict_name);

  if (iter != _metas.end()) {
    iter->second->cube_request_num.reset();
    iter->second->cube_rpcfail_num.reset();
  } else {
    LOG(WARNING) << "reset_bvar to invalid dict [" << dict_name << "]";
  }
}

MetaInfo* Meta::create_meta(const BUTIL_RAPIDJSON_NAMESPACE::Value& config) {
  if (!config.HasMember("dict_name") || !config.HasMember("shard") ||
      !config.HasMember("dup") || !config.HasMember("timeout") ||
      !config.HasMember("retry") || !config.HasMember("backup_request") ||
      !config.HasMember("type")) {
    LOG(ERROR) << "create meta failed, required fields miss";
    return NULL;
  }

  if (!config["dict_name"].IsString() || !config["shard"].IsInt() ||
      !config["dup"].IsInt() || !config["timeout"].IsInt() ||
      !config["retry"].IsInt() || !config["backup_request"].IsInt() ||
      !config["type"].IsString()) {
    LOG(ERROR) << "create meta failed, required fields type error";
    return NULL;
  }

  MetaInfo* info = new MetaInfo();
  info->dict_name = config["dict_name"].GetString();
  info->shard_num = config["shard"].GetInt();
  info->dup_num = config["dup"].GetInt();

  int timeout = config["timeout"].GetInt();
  int retry = config["retry"].GetInt();
  int backup_request = config["backup_request"].GetInt();
  std::string type = config["type"].GetString();
  std::string load_balancer = "rr";

  if (config.HasMember("load_balancer") && config["load_balancer"].IsString()) {
    load_balancer = config["load_balancer"].GetString();
  }

  int ret = 0;

  if (type.compare("ipport") == 0) {
    ret = create_meta_from_ipport(
        info, config["nodes"], timeout, retry, backup_request);
  } else if (type.compare("ipport_list") == 0) {
    ret = create_meta_from_ipport_list(
        info, config["nodes"], timeout, retry, backup_request, load_balancer);
  } else if (type.compare("ipport_parallel") == 0) {
    ret = create_meta_from_ipport_parallel(
        info, config["nodes"], timeout, retry, backup_request);
  } else {
    ret = -1;
  }

  if (ret != 0) {
    LOG(ERROR) << "create meta failed error=" << ret;
    delete info;
    return NULL;
  }

  return info;
}

int Meta::create_meta_from_ipport(MetaInfo* meta,
                                  const BUTIL_RAPIDJSON_NAMESPACE::Value& nodes,
                                  int timeout,
                                  int retry,
                                  int backup_request) {
  brpc::ChannelOptions options;
  options.timeout_ms = timeout;
  options.max_retry = retry;

  if (backup_request > 0) {
    options.backup_request_ms = backup_request;
  }

  meta->cube_conn.resize(meta->shard_num, NULL);

  for (int i = 0; i < static_cast<int>(nodes.Size()); ++i) {
    const BUTIL_RAPIDJSON_NAMESPACE::Value& node = nodes[i];
    const std::string& node_ip = node["ip"].GetString();
    int node_port = node["port"].GetInt();

    ::brpc::Channel* chan = new ::brpc::Channel();

    if (chan->Init(node_ip.c_str(), node_port, &options) != 0) {
      LOG(ERROR) << "Init connection to [" << node_ip << ":" << node_port
                 << "] failed";
      delete chan;
      return -1;
    }

    meta->cube_conn[i] = chan;
  }

  return 0;
}

int Meta::create_meta_from_ipport_list(
    MetaInfo* meta,
    const BUTIL_RAPIDJSON_NAMESPACE::Value& nodes,
    int timeout,
    int retry,
    int backup_request,
    const std::string& load_balancer) {
  brpc::ChannelOptions options;
  options.timeout_ms = timeout;
  options.max_retry = retry;

  if (backup_request > 0) {
    options.backup_request_ms = backup_request;
  }

  meta->cube_conn.resize(meta->shard_num, NULL);

  for (int i = 0; i < static_cast<int>(nodes.Size()); ++i) {
    const BUTIL_RAPIDJSON_NAMESPACE::Value& node = nodes[i];
    const std::string& ipport_list = node["ipport_list"].GetString();

    ::brpc::Channel* chan = new ::brpc::Channel();

    if (chan->Init(ipport_list.c_str(), load_balancer.c_str(), &options) != 0) {
      LOG(ERROR) << "Init connection to [" << ipport_list << "] failed";
      delete chan;
      return -1;
    }

    meta->cube_conn[i] = chan;
  }

  return 0;
}

int Meta::create_meta_from_ipport_parallel(
    MetaInfo* meta,
    const BUTIL_RAPIDJSON_NAMESPACE::Value& nodes,
    int timeout,
    int retry,
    int backup_request) {
  if (nodes.Size() < 1) {
    LOG(ERROR) << "Config nodes size less than 0";
    return -1;
  }

  brpc::ChannelOptions options;
  options.timeout_ms = timeout;
  options.max_retry = retry;

  if (backup_request > 0) {
    options.backup_request_ms = backup_request;
  }

  meta->cube_conn.resize(meta->shard_num, NULL);

  for (int i = 0; i < meta->shard_num; ++i) {
    const BUTIL_RAPIDJSON_NAMESPACE::Value& node = nodes[0];
    const std::string& node_ip = node["ip"].GetString();
    int node_port = node["port"].GetInt();

    ::brpc::Channel* chan = new ::brpc::Channel();

    if (chan->Init(node_ip.c_str(), node_port, &options) != 0) {
      LOG(ERROR) << "Init connection to [" << node_ip << ":" << node_port
                 << "] failed";
      delete chan;
      return -1;
    }

    meta->cube_conn[i] = chan;
  }

  return 0;
}

const std::vector<const MetaInfo*> Meta::metas() {
  std::vector<const MetaInfo*> metas;

  for (auto i : _metas) {
    metas.push_back(i.second);
  }

  return metas;
}

}  // namespace mcube
}  // namespace rec
