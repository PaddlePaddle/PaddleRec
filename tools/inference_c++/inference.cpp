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
#include "./criteo_reader.h"
#include "./safe_queue.h"
#include "paddle_inference/paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

DEFINE_string(data_dir, "", "Directory of the data.");
DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(params_file, "", "Directory of the inference parameters.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_bool(use_gpu, false, "use gpu.");
DEFINE_int32(batch_size, 1, "Directory of the inference model.");
DEFINE_int32(num_infer_threads, 1, "infer thread pools number");
DEFINE_int32(num_reader_threads, 1, "reader thread pools number");

// volatile bool is_producer_stop = false;
extern std::atomic<int> feeder_stops(0);

// 初始化Predictor
std::shared_ptr<Predictor> InitPredictor() {
  paddle_infer::Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_params_file);
  config.pass_builder()->DeletePass("repeated_fc_relu_fuse_pass");
  if (FLAGS_use_gpu) {
    config.EnableUseGpu(1000, 0);
  }
  return paddle_infer::CreatePredictor(config);
}

// 声明reader函数
extern void FeederProcess(std::string path, SharedQueue<click_joint> &queue);

// 定义预测函数
void InferenceProcess(SharedQueue<click_joint> &queue) {
  auto predictor = InitPredictor();
  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  while (!(feeder_stops == FLAGS_num_reader_threads && queue.empty())) {
    // 组batch，将batchsize条数据汇聚为一条数据。若reader已结束不会继续产生元素且队列中剩余元素已不足一个batch，则丢掉所有剩余元素。
    click_joint batch_data;
    if (feeder_stops == FLAGS_num_reader_threads &&
        queue.size() < FLAGS_batch_size) {
      int queue_num = queue.size();
      for (int drop = 0; drop < queue_num; drop++) {
        click_joint drop_last = queue.pop();
      }
      break;
    }
    for (int batch_id = 0; batch_id < FLAGS_batch_size; batch_id++) {
      click_joint one_data = queue.pop();
      batch_data.label.insert(
          batch_data.label.end(), one_data.label.begin(), one_data.label.end());
      batch_data.name_type = one_data.name_type;
      for (int i = 0; i < input_names.size(); i++) {
        std::string type = one_data.name_type[input_names[i]];
        if (type == "int") {
          batch_data.int_feasign[input_names[i]].insert(
              batch_data.int_feasign[input_names[i]].end(),
              one_data.int_feasign[input_names[i]].begin(),
              one_data.int_feasign[input_names[i]].end());
        }
        if (type == "float") {
          batch_data.float_feasign[input_names[i]].insert(
              batch_data.float_feasign[input_names[i]].end(),
              one_data.float_feasign[input_names[i]].begin(),
              one_data.float_feasign[input_names[i]].end());
        }
      }
    }
    // 根据名称获得slot对应的数据类型，并从队列中按照名称取出相应类型的数据，传入模型中预测
    for (int i = 0; i < input_names.size(); i++) {
      auto input_tensor = predictor->GetInputHandle(input_names[i]);
      std::string type = batch_data.name_type[input_names[i]];
      if (type == "int") {
        std::vector<int> input_shape = {
            FLAGS_batch_size,
            static_cast<int>(batch_data.int_feasign[input_names[i]].size()) /
                FLAGS_batch_size};
        input_tensor->Reshape(input_shape);
        input_tensor->CopyFromCpu(
            batch_data.int_feasign[input_names[i]].data());
      }
      if (type == "float") {
        std::vector<int> input_shape = {
            FLAGS_batch_size,
            static_cast<int>(batch_data.float_feasign[input_names[i]].size()) /
                FLAGS_batch_size};
        input_tensor->Reshape(input_shape);
        input_tensor->CopyFromCpu(
            batch_data.float_feasign[input_names[i]].data());
      }
    }

    // 预测
    predictor->Run();

    // 获取预测结果并输出
    for (int i = 0; i < output_names.size(); i++) {
      auto output_tensor = predictor->GetOutputHandle(output_names[i]);
      std::vector<float> output_data;
      std::vector<int> output_shape = output_tensor->shape();
      int out_num = std::accumulate(
          output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
      output_data.resize(out_num);
      output_tensor->CopyToCpu(output_data.data());
      std::cout << "[ ";
      for (int j = 0; j < FLAGS_batch_size; j++) {
        std::cout << output_data[j] << " ";
      }
      std::cout << "]" << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  // 从reader中读取的数据将放入queue队列中，队列中每一个元素即为一条样本
  SharedQueue<click_joint> queue;
  // 建立多个reader线程，每条reader线程独立的从文件中解析数据并将数据放入queue队列中
  std::vector<std::thread> feeder_threads;
  for (int i = 0; i < FLAGS_num_reader_threads; i++) {
    std::thread feeder_thread(FeederProcess, FLAGS_data_dir, std::ref(queue));
    feeder_thread.detach();
    feeder_threads.push_back(std::move(feeder_thread));
  }
  // 建立多个预测线程,每条预测线程独立的从queue队列中获取数据并执行预测。
  std::vector<std::thread> infer_threads;
  for (int j = 0; j < FLAGS_num_infer_threads; j++) {
    std::thread infer_thread(InferenceProcess, std::ref(queue));
    infer_thread.detach();
    infer_threads.push_back(std::move(infer_thread));
  }
  // 当reader线程还未完全结束，或者queue队列中还有数据没有被预测，那么主线程会等待预测结束。
  while (!(feeder_stops == FLAGS_num_reader_threads && queue.empty())) {
    sleep(1);
  }
  return 0;
}
