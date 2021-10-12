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

#include "../include/cube_api.h"
#ifdef BCLOUD
#include <baidu/rpc/channel.h>
#include <baidu/rpc/parallel_channel.h>
#else
#include <brpc/channel.h>
#include <brpc/parallel_channel.h>
#endif

#include <google/protobuf/descriptor.h>
#include "../include/cube_api_bvar.h"

#include "../include/error.h"
#include "../include/meta.h"
#include "butil/logging.h"
#include "brpc/server.h"
DECLARE_string(config_file);
namespace {
static ::rec::mcube::CubeAPI* g_ins = NULL;
}

#ifdef BCLOUD
namespace brpc = baidu::rpc;
#endif

namespace rec {
namespace mcube {

struct DictRpcData {
  std::vector<DictRequest> sub_reqs;
  std::vector<DictResponse> sub_res;
};

static void dict_rpc_deleter(void* d) { delete static_cast<DictRpcData*>(d); }

static void sub_seek_done(DictResponse* response,
                          brpc::Controller* cntl,
                          std::vector<int>* offset,
                          std::function<void(DictValue*, size_t)> parse) {
  // std::unique_ptr<DictResponse> response_guard(response);
  // std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  if (cntl->Failed()) {
    for (int i = 0; i < response->values().size(); ++i) {
      DictValue* val = response->mutable_values(i);
      val->set_status(CubeError::E_SEEK_FAILED);
      *val->mutable_value() = "";
      parse(val, (*offset)[i]);
    }
  } else {
    for (int i = 0; i < response->values().size(); ++i) {
      DictValue* val = response->mutable_values(i);
      parse(val, (*offset)[i]);
    }
  }
}

struct DoNothing : public google::protobuf::Closure {
  void Run() {}
};

CubeAPI* CubeAPI::instance() {
  if (g_ins == NULL) {
    g_ins = new CubeAPI();
  }

  return g_ins;
}

void CubeAPI::cleanup() {
  if (g_ins != NULL) {
    g_ins->destroy();
    delete g_ins;
    g_ins = NULL;
  }
}

CubeAPI::CubeAPI() : _meta(NULL) 
{
	_meta = Meta::instance();
  int ret = _meta->init(FLAGS_config_file.c_str());

  if (ret != 0) {
    //LOG(ERROR) << "Init cube meta failed";
  }

  CHECK_EQ(0, bthread_key_create(&_tls_key, dict_rpc_deleter));
}
CubeAPI::~CubeAPI() { CHECK_EQ(0, bthread_key_delete(_tls_key)); }

int CubeAPI::init(const char* conf_file) {
  // init meta
/*
  _meta = Meta::instance();
  int ret = _meta->init(conf_file);

  if (ret != 0) {
    //LOG(ERROR) << "Init cube meta failed";
    return ret;
  }

  CHECK_EQ(0, bthread_key_create(&_tls_key, dict_rpc_deleter));
*/
  return 0;
}

int CubeAPI::destroy() {
  // Meta* meta = Meta::instance();
  if (_meta == NULL) {
    //LOG(WARNING) << "Destroy, cube meta is null";
    return 0;
  }

  int ret = _meta->destroy();

  if (ret != 0) {
    //LOG(WARNING) << "Destroy cube meta failed";
  }

  return 0;
}

int CubeAPI::seek(const std::string& dict_name,
                  const uint64_t& key,
                  CubeValue* val) {
  if (_meta == NULL) {
    //LOG(ERROR) << "seek, meta is null";
    return -1;
  }

  MetaInfo* info = _meta->get_meta(dict_name);

  if (info == NULL) {
    //LOG(ERROR) << "get meta [" << dict_name << "] failed";
    return -1;
  }

  int shard_id = key % info->shard_num;
  DictRequest req;
  DictResponse res;

  req.add_keys(key);

  ::brpc::Channel* chan = info->cube_conn[shard_id];

  DictService_Stub* stub = new DictService_Stub(chan);
  brpc::Controller* cntl = new ::brpc::Controller();

  stub->seek(cntl, &req, &res, NULL);

  int ret = CubeError::E_OK;
  if (cntl->Failed()) {
    info->cube_rpcfail_num << 1;

    val->error = CubeError::E_SEEK_FAILED;
    val->buff.assign("");

    ret = CubeError::E_ALL_SEEK_FAILED;
    std::cout << "cube seek from shard [" << shard_id << "] failed ["
                 << cntl->ErrorText() << "]";
  } else if (res.values().size() > 0) {
    DictValue* res_val = res.mutable_values(0);
    if (static_cast<int>(res_val->status()) == CubeError::E_NO_SUCH_KEY) {
      g_cube_keys_miss_num << 1;
    }
    val->error = res_val->status();
    val->buff.swap(*res_val->mutable_value());
  } else {
    val->error = CubeError::E_SEEK_FAILED;
    val->buff.assign("");
    ret = CubeError::E_ALL_SEEK_FAILED;
  }
  info->cube_request_num << 1;
  g_cube_keys_num << 1;

  // cleanup
  delete stub;
  stub = NULL;
  delete cntl;
  cntl = NULL;

  return ret;
}

int CubeAPI::seek(const std::string& dict_name,
                  const std::vector<uint64_t>& keys,
                  std::vector<CubeValue>* vals) {
  // Meta* meta = Meta::instance();
  if (_meta == NULL) {
    //LOG(ERROR) << "seek, meta is null";
    return -1;
  }

  MetaInfo* info = _meta->get_meta(dict_name);

  if (info == NULL) {
    LOG(ERROR) << "get meta [" << dict_name << "] failed";
    return -1;
  }

  int shard_num = info->shard_num;

  DictRpcData* rpc_data =
      static_cast<DictRpcData*>(bthread_getspecific(_tls_key));

  if (rpc_data == NULL) {
    rpc_data = new DictRpcData;
    CHECK_EQ(0, bthread_setspecific(_tls_key, rpc_data));
  }

  rpc_data->sub_reqs.resize(shard_num);
  rpc_data->sub_res.resize(shard_num);

  std::vector<std::vector<int>> offset;
  offset.resize(shard_num);
  int init_cnt = keys.size() * 2 / shard_num;

  for (int i = 0; i < shard_num; ++i) {
    offset[i].reserve(init_cnt);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t shard_id = keys[i] % shard_num;
    rpc_data->sub_reqs[shard_id].add_keys(keys[i]);
    offset[shard_id].push_back(i);
  }

  std::vector<DictService_Stub*> stubs(shard_num);
  std::vector<brpc::Controller*> cntls(shard_num);

  for (int i = 0; i < shard_num; ++i) {
    ::brpc::Channel* chan = info->cube_conn[i];
    stubs[i] = new DictService_Stub(chan);
    cntls[i] = new ::brpc::Controller();
  }

  DoNothing do_nothing;

  for (int i = 0; i < shard_num; ++i) {
    stubs[i]->seek(
        cntls[i], &rpc_data->sub_reqs[i], &rpc_data->sub_res[i], &do_nothing);
  }

  int cntls_failed_cnt = 0;

  for (int i = 0; i < shard_num; ++i) {
    brpc::Join(cntls[i]->call_id());

    if (cntls[i]->Failed()) {
      ++cntls_failed_cnt;
      std::cout << "cube seek from shard [" << i << "] failed ["
                   << cntls[i]->ErrorText() << "]";
    }
  }

  int ret = CubeError::E_OK;

  info->cube_request_num << 1;

  if (cntls_failed_cnt > 0) {
    info->cube_rpcfail_num << 1;
    if (cntls_failed_cnt == shard_num) {
      ret = CubeError::E_ALL_SEEK_FAILED;
    } else {
      ret = CubeError::E_SEEK_FAILED;
    }
  }

  vals->resize(keys.size());

  // merge
  size_t miss_cnt = 0;
  for (int i = 0; i < shard_num; ++i) {
    if (cntls[i]->Failed()) {
      for (int j = 0; j < rpc_data->sub_res[i].values().size(); ++j) {
        (*vals)[offset[i][j]].error = CubeError::E_SEEK_FAILED;
        (*vals)[offset[i][j]].buff.assign("");
      }
    } else {
      for (int j = 0; j < rpc_data->sub_res[i].values().size(); ++j) {
        DictValue* val = rpc_data->sub_res[i].mutable_values(j);
        if (static_cast<int>(val->status()) == CubeError::E_NO_SUCH_KEY) {
          miss_cnt += 1;
        }
        (*vals)[offset[i][j]].error = val->status();
        (*vals)[offset[i][j]].buff.swap(*val->mutable_value());
      }
    }
  }

  // bvar stats
  g_cube_keys_num << keys.size();
  if (keys.size() > 0) {
    g_cube_keys_miss_num << miss_cnt;
    g_cube_value_size << (*vals)[0].buff.size();
  }

  // cleanup
  for (int i = 0; i < shard_num; ++i) {
    delete stubs[i];
    stubs[i] = NULL;
    delete cntls[i];
    cntls[i] = NULL;
    rpc_data->sub_reqs[i].Clear();
    rpc_data->sub_res[i].Clear();
  }

  return ret;
}

int CubeAPI::opt_seek(const std::string& dict_name,
                      const std::vector<uint64_t>& keys,
                      std::function<void(DictValue*, size_t)> parse) {
  if (_meta == NULL) {
    //LOG(ERROR) << "seek, meta is null";
    return -1;
  }

  MetaInfo* info = _meta->get_meta(dict_name);

  if (info == NULL) {
    //LOG(ERROR) << "get meta [" << dict_name << "] failed";
    return -1;
  }

  int shard_num = info->shard_num;

  DictRpcData* rpc_data =
      static_cast<DictRpcData*>(bthread_getspecific(_tls_key));

  if (rpc_data == NULL) {
    rpc_data = new DictRpcData;
    CHECK_EQ(0, bthread_setspecific(_tls_key, rpc_data));
  }

  rpc_data->sub_reqs.resize(shard_num);
  rpc_data->sub_res.resize(shard_num);

  std::vector<std::vector<int>> offset;
  offset.resize(shard_num);
  int init_cnt = keys.size() * 2 / shard_num;

  for (int i = 0; i < shard_num; ++i) {
    offset[i].reserve(init_cnt);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t shard_id = keys[i] % shard_num;
    rpc_data->sub_reqs[shard_id].add_keys(keys[i]);
    offset[shard_id].push_back(i);
  }

  std::vector<DictService_Stub*> stubs(shard_num);
  std::vector<brpc::Controller*> cntls(shard_num);

  for (int i = 0; i < shard_num; ++i) {
    ::brpc::Channel* chan = info->cube_conn[i];
    stubs[i] = new DictService_Stub(chan);
    cntls[i] = new ::brpc::Controller();
  }

  for (int i = 0; i < shard_num; ++i) {
    stubs[i]->seek(cntls[i],
                   &rpc_data->sub_reqs[i],
                   &rpc_data->sub_res[i],
                   brpc::NewCallback(sub_seek_done,
                                     &rpc_data->sub_res[i],
                                     cntls[i],
                                     &(offset[i]),
                                     parse));
  }

  int cntls_failed_cnt = 0;

  for (int i = 0; i < shard_num; ++i) {
    brpc::Join(cntls[i]->call_id());

    if (cntls[i]->Failed()) {
      ++cntls_failed_cnt;
      std::cout << "cube seek from shard [" << i << "] failed ["
                   << cntls[i]->ErrorText() << "]";
    }
  }

  int ret = CubeError::E_OK;

  info->cube_request_num << 1;

  if (cntls_failed_cnt > 0) {
    info->cube_rpcfail_num << 1;
    if (cntls_failed_cnt == shard_num) {
      ret = CubeError::E_ALL_SEEK_FAILED;
    } else {
      ret = CubeError::E_SEEK_FAILED;
    }
  }

  // merge
  size_t miss_cnt = 0;
  for (int i = 0; i < shard_num; ++i) {
    if (!cntls[i]->Failed()) {
      for (int j = 0; j < rpc_data->sub_res[i].values().size(); ++j) {
        if (static_cast<int>(rpc_data->sub_res[i].values(j).status()) ==
            CubeError::E_NO_SUCH_KEY) {
          ++miss_cnt;
        }
      }
    }
  }

  // bvar stats
  g_cube_keys_num << keys.size();
  if (keys.size() > 0) {
    g_cube_keys_miss_num << miss_cnt;
  }

  // cleanup
  for (int i = 0; i < shard_num; ++i) {
    delete stubs[i];
    stubs[i] = NULL;
    delete cntls[i];
    cntls[i] = NULL;
    rpc_data->sub_reqs[i].Clear();
    rpc_data->sub_res[i].Clear();
  }

  return ret;
}

int CubeAPI::seek(const std::string& dict_name,
                  const std::vector<uint64_t>& keys,
                  std::vector<CubeValue>* vals,
                  std::string* version) {
  // Meta* meta = Meta::instance();
  if (_meta == NULL) {
    //LOG(ERROR) << "seek, meta is null";
    return -1;
  }

  MetaInfo* info = _meta->get_meta(dict_name);

  if (info == NULL) {
    //LOG(ERROR) << "get meta [" << dict_name << "] failed";
    return -1;
  }

  int shard_num = info->shard_num;

  DictRpcData* rpc_data =
      static_cast<DictRpcData*>(bthread_getspecific(_tls_key));

  if (rpc_data == NULL) {
    rpc_data = new DictRpcData;
    CHECK_EQ(0, bthread_setspecific(_tls_key, rpc_data));
  }

  rpc_data->sub_reqs.resize(shard_num);
  rpc_data->sub_res.resize(shard_num);

  std::vector<std::vector<int>> offset;
  offset.resize(shard_num);
  int init_cnt = keys.size() * 2 / shard_num;

  for (int i = 0; i < shard_num; ++i) {
    offset[i].reserve(init_cnt);
    rpc_data->sub_reqs[i].set_version_required(true);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t shard_id = keys[i] % shard_num;
    rpc_data->sub_reqs[shard_id].add_keys(keys[i]);
    offset[shard_id].push_back(i);
  }

  std::vector<DictService_Stub*> stubs(shard_num);
  std::vector<brpc::Controller*> cntls(shard_num);

  for (int i = 0; i < shard_num; ++i) {
    ::brpc::Channel* chan = info->cube_conn[i];
    stubs[i] = new DictService_Stub(chan);
    cntls[i] = new ::brpc::Controller();
  }

  DoNothing do_nothing;

  for (int i = 0; i < shard_num; ++i) {
    stubs[i]->seek(
        cntls[i], &rpc_data->sub_reqs[i], &rpc_data->sub_res[i], &do_nothing);
  }

  int cntls_failed_cnt = 0;

  for (int i = 0; i < shard_num; ++i) {
    brpc::Join(cntls[i]->call_id());

    if (cntls[i]->Failed()) {
      ++cntls_failed_cnt;
      std::cout << "cube seek from shard [" << i << "] failed ["
                   << cntls[i]->ErrorText() << "]";
    }
  }

  int ret = CubeError::E_OK;
  info->cube_request_num << 1;

  if (cntls_failed_cnt > 0) {
    info->cube_rpcfail_num << 1;
    if (cntls_failed_cnt == shard_num) {
      ret = CubeError::E_ALL_SEEK_FAILED;
    } else {
      ret = CubeError::E_SEEK_FAILED;
    }
  }

  vals->resize(keys.size());

  // merge
  size_t miss_cnt = 0;
  for (int i = 0; i < shard_num; ++i) {
    if (cntls[i]->Failed()) {
      for (int j = 0; j < rpc_data->sub_res[i].values().size(); ++j) {
        (*vals)[offset[i][j]].error = CubeError::E_SEEK_FAILED;
        (*vals)[offset[i][j]].buff.assign("");
      }
    } else {
      for (int j = 0; j < rpc_data->sub_res[i].values().size(); ++j) {
        DictValue* val = rpc_data->sub_res[i].mutable_values(j);
        if (static_cast<int>(val->status()) == CubeError::E_NO_SUCH_KEY) {
          miss_cnt += 1;
        }
        (*vals)[offset[i][j]].error = val->status();
        (*vals)[offset[i][j]].buff.swap(*val->mutable_value());
      }
      if (version->compare(rpc_data->sub_res[i].version()) < 0) {
        *version = rpc_data->sub_res[i].version();
      }
    }
  }

  // bvar stats
  g_cube_keys_num << keys.size();
  if (keys.size() > 0) {
    g_cube_keys_miss_num << miss_cnt;
    g_cube_value_size << (*vals)[0].buff.size();
  }

  // cleanup
  for (int i = 0; i < shard_num; ++i) {
    delete stubs[i];
    stubs[i] = NULL;
    delete cntls[i];
    cntls[i] = NULL;
    rpc_data->sub_reqs[i].Clear();
    rpc_data->sub_res[i].Clear();
  }

  return ret;
}

int CubeAPI::opt_seek(const std::string& dict_name,
                      const std::vector<uint64_t>& keys,
                      std::function<void(DictValue*, size_t)> parse,
                      std::string* version) {
  if (_meta == NULL) {
    //LOG(ERROR) << "seek, meta is null";
    return -1;
  }

  MetaInfo* info = _meta->get_meta(dict_name);

  if (info == NULL) {
    //LOG(ERROR) << "get meta [" << dict_name << "] failed";
    return -1;
  }

  int shard_num = info->shard_num;

  DictRpcData* rpc_data =
      static_cast<DictRpcData*>(bthread_getspecific(_tls_key));

  if (rpc_data == NULL) {
    rpc_data = new DictRpcData;
    CHECK_EQ(0, bthread_setspecific(_tls_key, rpc_data));
  }

  rpc_data->sub_reqs.resize(shard_num);
  rpc_data->sub_res.resize(shard_num);

  std::vector<std::vector<int>> offset;
  offset.resize(shard_num);
  int init_cnt = keys.size() * 2 / shard_num;

  for (int i = 0; i < shard_num; ++i) {
    offset[i].reserve(init_cnt);
    rpc_data->sub_reqs[i].set_version_required(true);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t shard_id = keys[i] % shard_num;
    rpc_data->sub_reqs[shard_id].add_keys(keys[i]);
    offset[shard_id].push_back(i);
  }

  std::vector<DictService_Stub*> stubs(shard_num);
  std::vector<brpc::Controller*> cntls(shard_num);

  for (int i = 0; i < shard_num; ++i) {
    ::brpc::Channel* chan = info->cube_conn[i];
    stubs[i] = new DictService_Stub(chan);
    cntls[i] = new ::brpc::Controller();
  }

  for (int i = 0; i < shard_num; ++i) {
    stubs[i]->seek(cntls[i],
                   &rpc_data->sub_reqs[i],
                   &rpc_data->sub_res[i],
                   brpc::NewCallback(sub_seek_done,
                                     &rpc_data->sub_res[i],
                                     cntls[i],
                                     &(offset[i]),
                                     parse));
  }

  int cntls_failed_cnt = 0;

  for (int i = 0; i < shard_num; ++i) {
    brpc::Join(cntls[i]->call_id());

    if (cntls[i]->Failed()) {
      ++cntls_failed_cnt;
      std::cout << "cube seek from shard [" << i << "] failed ["
                   << cntls[i]->ErrorText() << "]";
    }
  }

  int ret = CubeError::E_OK;

  info->cube_request_num << 1;

  if (cntls_failed_cnt > 0) {
    info->cube_rpcfail_num << 1;
    if (cntls_failed_cnt == shard_num) {
      ret = CubeError::E_ALL_SEEK_FAILED;
    } else {
      ret = CubeError::E_SEEK_FAILED;
    }
  }

  // merge
  size_t miss_cnt = 0;
  for (int i = 0; i < shard_num; ++i) {
    if (!cntls[i]->Failed()) {
      for (int j = 0; j < rpc_data->sub_res[i].values().size(); ++j) {
        if (static_cast<int>(rpc_data->sub_res[i].values(j).status()) ==
            CubeError::E_NO_SUCH_KEY) {
          ++miss_cnt;
        }
      }
      if (version->compare(rpc_data->sub_res[i].version()) < 0) {
        *version = rpc_data->sub_res[i].version();
      }
    }
  }

  // bvar stats
  g_cube_keys_num << keys.size();
  if (keys.size() > 0) {
    g_cube_keys_miss_num << miss_cnt;
  }

  // cleanup
  for (int i = 0; i < shard_num; ++i) {
    delete stubs[i];
    stubs[i] = NULL;
    delete cntls[i];
    cntls[i] = NULL;
    rpc_data->sub_reqs[i].Clear();
    rpc_data->sub_res[i].Clear();
  }

  return ret;
}

std::vector<std::string> CubeAPI::get_table_names() {
  const std::vector<const MetaInfo*> metas = _meta->metas();
  std::vector<std::string> table_names;
  for (auto itr = metas.begin(); itr != metas.end(); ++itr) {
    table_names.push_back((*itr)->dict_name);
  }
  return table_names;
}
}  // namespace mcube
}  // namespace rec
