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

#include <gflags/gflags.h>
#include <algorithm>
#include <atomic>
#include <stdio.h>
#include <fstream>
#include <thread> //NOLINT
#include "cube_api.h"
#include "utils.h"
#include "xbox_pb_deconverter.h"

#define TIME_FLAG(flag)  \
	struct timeval flag; \
	gettimeofday(&(flag), NULL);
extern std::vector<uint64_t> globalKeys;
DECLARE_bool(debug);
DEFINE_string(config_file, "../cube_app/conf/cube.conf", "m-cube config file");
DECLARE_string(keys);
DEFINE_string(dict, "test_dict", "dict to seek");
DEFINE_int32(timeout, 200, "timeout in ms");
DEFINE_int32(retry, 3, "retry times");
DEFINE_int32(cube_thread_num, 1, "thread num");
DEFINE_string(cube_result_file, "../cube.result", "cube qry result");

std::atomic<int> g_concurrency(0);

namespace
{
	inline uint64_t time_diff(const struct timeval &start_time,
							  const struct timeval &end_time)
	{
		return (end_time.tv_sec - start_time.tv_sec) * 1000000 +
			   (end_time.tv_usec - start_time.tv_usec);
	}
} // namespace

namespace rec
{
	namespace mcube
	{
		std::string string_to_hex(const std::string &input)
		{
			static const char *const lut = "0123456789ABCDEF";
			size_t len = input.length();

			std::string output;
			output.reserve(2 * len);
			for (size_t i = 0; i < len; ++i)
			{
				const unsigned char c = input[i];
				output.push_back(lut[c >> 4]);
				output.push_back(lut[c & 15]);
			}
			return output;
		}

		int run(std::unordered_map<uint64_t, std::vector<float>>& result, int thread_id, std::set<uint64_t>& feasigns)
		{
			CubeAPI *cube = CubeAPI::instance();
			int ret = 0;
			//ret = cube->init(FLAGS_config_file.c_str());
			if (ret != 0) {
				std::cout << "init cube api failed err=" << ret;
				return ret;
			}
			std::vector<std::string> table_names = cube->get_table_names();
			if (table_names.empty()) {
				LOG(ERROR) << "cube table_names is null!";
			}
			std::vector<CubeValue> values;
			std::vector<uint64_t> keys;
			//keys.assign(feasigns.begin(), feasigns.end());
			if (feasigns.size() > globalKeys.size()) {
				LOG(ERROR) << "global keys not enough!";
			}
			keys.assign(globalKeys.begin(), globalKeys.begin() + feasigns.size());
			/*
			std::ifstream keyFile(FLAGS_keys.c_str());
			if (!keyFile.is_open()) {
				LOG(ERROR) << "cube key file open failed!";
			}
			std::string line;
			while (getline(keyFile, line)) {
				keys.push_back(std::stoul(line));
			}
			*/
			/*
			if (keys.size() > FLAGS_cube_batch) {
				LOG(WARNING) << "keys size exceed cube batch size!";
			}
			*/
			if (keys.size() == 0) {
				LOG(ERROR) << "key list is null";
				return 0;
			}
			while (g_concurrency.load() >= FLAGS_cube_thread_num) {
			}
			g_concurrency++;

			TIME_FLAG(seek_start);
			ret = cube->seek(FLAGS_dict, keys, &values);
			TIME_FLAG(seek_end);
			if (ret != 0) {
				LOG(ERROR) << "cube seek failed";
			} 
			if (values.size() != keys.size() || values[0].buff.size() == 0) {
				LOG(ERROR) << "keys.size(): " << keys.size() << " values.size(): " << values.size() << " cube value return null";
			}
			uint64_t seek_cost = time_diff(seek_start, seek_end);
			// convert && save cube qurey result
			paddleRecInfer::xbox_pb_converter::XboxPbDeconverter deconverter;
			for (uint64_t i = 0; i < keys.size(); i++) {
				if (result.find(keys[i]) != result.end()) {
					continue;
				}
				deconverter.Deconvert(keys[i], values[i].buff);
				result[keys[i]].swap(deconverter.mf);
			}
			keys.clear();
			values.clear();

			g_concurrency--;
			//keyFile.close();
			//ret = cube->destroy();
			//if (ret != 0) {
			//	LOG(ERROR) << "destroy cube api failed err=" << ret;
			//}
			return 0;
		}

		int run_m(std::set<uint64_t>& keys, std::unordered_map<uint64_t, std::vector<float>>& queryResult)
		{
			//LOG(INFO) << "enter qurey cube main ...";
			std::vector<std::thread *> thread_pool;
			std::unordered_map<uint64_t, std::unordered_map<uint64_t, std::vector<float>>> result; // key1: threadId, key2: feasign
			for (int i = 0; i < FLAGS_cube_thread_num; i++) {
				thread_pool.push_back(new std::thread(run, std::ref(result[i]), i, std::ref(keys)));
			}
			for (int i = 0; i < FLAGS_cube_thread_num; i++) {
				thread_pool[i]->join();
				delete thread_pool[i];
			}
			for (auto it1 = result.begin(); it1 != result.end(); it1++) {
				for (auto it2 = result[it1->first].begin(); it2 != result[it1->first].end(); it2++) {
					queryResult[it2->first].swap(it1->second[it2->first]);
				}
			}
			// only for one cube query thread
			//for (auto it = result[0].begin(); it != result[0].end(); it++) {
			//    queryResult[it->first] = std::move(it->second);
			//}
			//LOG(INFO) << "exit query cube main ...";
			return 0;
		}

	} // namespace mcube
} // namespace rec
