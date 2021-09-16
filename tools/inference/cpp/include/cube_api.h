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

#include <stdint.h>
#include <string>
#include <vector>

#ifdef BCLOUD
#include "baidu/rpc/server.h"
#else
//#include "brpc/server.h"
#endif

#include "cube.pb.h"
#include "meta.h"

namespace rec
{
	namespace mcube
	{

		struct CubeValue
		{
			int error;
			std::string buff;
		};

		class CubeAPI
		{
		public:
			static CubeAPI *instance();

			static void cleanup();

		public:
			CubeAPI();
			~CubeAPI();

			/**
   * @brief: init api, not thread safe, should be called before seek.
   * @param [in] conf_file: filepath of config file.
   * @return: error code. 0 for succ.
   */
			int init(const char *conf_file);

			/**
   * @brief: destroy api, should be called before program exit.
   * @return: error code. 0 for succ.
   */
			int destroy();

			/** brief: get data from cube server, thread safe.
   * @param [in] dict_name: dict name.
   * @param [in] key: key to seek.
   * @param [out] val: value of key.
   * @return: error code. 0 for succ.
   */
			int seek(const std::string &dict_name, const uint64_t &key, CubeValue *val);

			/**
   * @brief: get data from cube server, thread safe.
   * @param [in] dict_name: dict name.
   * @param [in] keys: keys to seek.
   * @param [out] vals: value of keys.
   * @return: TODO
   */
			int seek(const std::string &dict_name,
					 const std::vector<uint64_t> &keys,
					 std::vector<CubeValue> *vals);

			int opt_seek(const std::string &dict_name,
						 const std::vector<uint64_t> &keys,
						 std::function<void(DictValue *, size_t)> parse);

			/**
   * @brief: get data from cube server, thread safe.
   * @param [in] dict_name: dict name.
   * @param [in] keys: keys to seek.
   * @param [out] vals: value of keys.
   * @param [out] version: data version.
   * @return: TODO
   */
			int seek(const std::string &dict_name,
					 const std::vector<uint64_t> &keys,
					 std::vector<CubeValue> *vals,
					 std::string *version);

			int opt_seek(const std::string &dict_name,
						 const std::vector<uint64_t> &keys,
						 std::function<void(DictValue *, size_t)> parse,
						 std::string *version);

			/**
   * @brief: get all table names from cube server, thread safe.
   * @param [out] vals: vector of table names
   *
   */
			std::vector<std::string> get_table_names();

		public:
			static const char *error_msg(int error_code);

		private:
			CubeAPI(const CubeAPI &) {}

		private:
			Meta *_meta;
			bthread_key_t _tls_key;
			// void split(const std::vector<uint64_t>& keys);
		};

	} // namespace mcube
} // namespace rec
