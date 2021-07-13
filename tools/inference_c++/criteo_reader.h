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

#pragma once
#include <dirent.h>
#include <gflags/gflags.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include "./safe_queue.h"

extern std::atomic<int> feeder_stops;
void split(const std::string& s,
           std::vector<std::string>& tokens,
           const std::string& delimiters = " ") {
  std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
  std::string::size_type pos = s.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delimiters, pos);
    pos = s.find_first_of(delimiters, lastPos);
  }
}

template <typename T>
std::vector<T> slice(std::vector<T>& v, int m, int n) {
  std::vector<T> vec(n - m);

  if (m == n) {
    return vec;
  }
  std::copy(v.begin() + m, v.begin() + n, vec.begin());
  return vec;
}

struct click_joint {
  std::vector<int> label;
  std::map<std::string, std::vector<int>> int_feasign;
  std::map<std::string, std::vector<float>> float_feasign;
  std::map<std::string, std::string> name_type;
};

void FeederProcess(std::string path, SharedQueue<click_joint>& queue) {
  // 获取文件列表
  unsigned char isFile = 0x8;
  std::vector<std::string> files;
  const char* data_path = path.data();
  DIR* dirptr = opendir(data_path);
  struct dirent* entry;
  while ((entry = readdir(dirptr)) != NULL) {
    if (entry->d_type == isFile) {
      files.push_back(entry->d_name);
    }
  }
  closedir(dirptr);
  // 从每一个文件中按行读取内容
  for (int i = 0; i < files.size(); i++) {  // 每个文件
    std::ifstream file;
    file.open(path + "/" + files[i]);
    std::string line;
    std::vector<std::string> paragraph;
    std::vector<std::string> slot_feasign;
    std::vector<std::string> sparse_slot = {
        "click", "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",
        "9",     "10", "11", "12", "13", "14", "15", "16", "17",
        "18",    "19", "20", "21", "22", "23", "24", "25", "26"};
    std::vector<std::string> dense_slot = {"dense_feature"};
    // 每一行
    while (getline(file, line)) {
      split(line, paragraph, " ");
      std::vector<std::string> slot;
      std::vector<float> dense_feasign;
      std::vector<int> sparse_feasign;
      // 每一个slot:feasign对
      for (int x = 0; x < paragraph.size(); x++) {
        split(paragraph[x], slot_feasign, ":");
        // 如果是sparse_slot,转int值放入sparse_feasign中
        if (find(sparse_slot.begin(), sparse_slot.end(), slot_feasign[0]) !=
            sparse_slot.end()) {
          slot.push_back(slot_feasign[0]);
          sparse_feasign.push_back(std::stoi(slot_feasign[1]));
        }
        // 如果是dense_slot,转float值放入dense_feasign中
        if (slot_feasign[0] == dense_slot[0]) {
          dense_feasign.push_back(std::stof(slot_feasign[1]));
        }
        slot_feasign.clear();
      }
      slot.push_back("dense_input");

      // 如果数据有缺漏，padding补齐
      if (dense_feasign.size() < 13 || sparse_feasign.size() < 27) {
        int padding = 0;
        if (dense_feasign.size() < 13) {
          for (int y = 0; y < (13 - dense_feasign.size()); y++) {
            dense_feasign.push_back(static_cast<float>(padding));
          }
        }
        if (sparse_feasign.size() < 27) {
          for (int y = 0; y < sparse_slot.size(); y++) {
            if (sparse_slot[y] != slot[y]) {
              slot.insert(slot.begin() + y, sparse_slot[y]);
              sparse_feasign.insert(sparse_feasign.begin() + y, padding);
            }
          }
        }
      }

      // 将dense_feasign,sparse_feasign都按照名称放入queue的一个节点中，并创建一个name和数据类型的对照表。
      click_joint sample;
      sample.label.push_back(sparse_feasign[0]);
      for (int x = 1; x < sparse_feasign.size(); x++) {
        std::vector<int> tmp = {sparse_feasign[x]};
        sample.int_feasign.insert(
            std::pair<std::string, std::vector<int>>("C" + slot[x], tmp));
        sample.name_type.insert(
            std::pair<std::string, std::string>("C" + slot[x], "int"));
      }
      sample.float_feasign.insert(std::pair<std::string, std::vector<float>>(
          slot[slot.size() - 1], dense_feasign));
      sample.name_type.insert(
          std::pair<std::string, std::string>(slot[slot.size() - 1], "float"));
      queue.push_back(sample);

      dense_feasign.clear();
      sparse_feasign.clear();
      slot.clear();
      paragraph.clear();
    }
    file.close();
  }
  feeder_stops.fetch_add(1);
}
