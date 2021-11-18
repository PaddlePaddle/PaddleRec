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

#ifndef BAIDU_PADDLEPADDLE_PADDLEREC_INFERENCE_H
#define BAIDU_PADDLEPADDLE_PADDLEREC_INFERENCE_H
#include <string>
#include <iostream>
#include <thread>
#include <fstream>
#include <set>
#include <vector>
#include <numeric>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <string.h>
#include <sys/time.h>
#include <queue>
#include <deque>
#include <unordered_map>
#include <map>
#include "paddle_inference_api.h"
#include "paddle_api.h"
#include "paddle_analysis_config.h"
#include "utils.h"
#include "cube_cli.h"

#ifdef CUBE
using dataType = float;
#else
using dataType = int64_t;
#endif
extern SharedQueue<BatchSample<dataType>> batchSamples;
extern std::vector<uint64_t> globalKeys;
DECLARE_bool(debug);
DECLARE_bool(withCube);
DECLARE_int32(threadNum);
DECLARE_int32(iterationNum);
DECLARE_int32(batchSize);
DECLARE_string(trainingFile);
DECLARE_string(performanceFile);
DECLARE_string(modelFile);
DECLARE_string(paramFile);
DECLARE_string(keys);
DECLARE_string(predictorLog);
DECLARE_string(stdLog);
DECLARE_string(varsName);

void WritePerformance(Metric& metric);

class Timer {
public:
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<float> duration;
    Timer()
    {
        start = std::chrono::high_resolution_clock::now();
    }
    ~Timer()
    {
        LOG(INFO) << "Timer deconsruct" << "\n";
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
    }
};

class PaddleInferModel {
public:
    PaddleInferModel() {};

public:
    void ReadConfig()
    {
        USE_GPU = 0;
        EMBEDDING_SIZE = 9;
        SLOT_NUMBER = 301;
        place = paddle::PaddlePlace::kCPU;
    }

    void LoadModel()
    {
        paddle_infer::Config config;
        if (!FLAGS_debug) {
            config.DisableGlogInfo();
        }
        config.SetModel(FLAGS_modelFile, FLAGS_paramFile);
        if(FLAGS_debug) {
            config.SwitchIrOptim(static_cast<bool>(false));
        } else {
            config.SwitchIrOptim(static_cast<bool>(true));
        }
        config.EnableMKLDNN();
        //config.SwitchSpecifyInputNames();
        config.SetCpuMathLibraryNumThreads(1);
        //config.pass_builder()->DeletePass("repeated_fc_relu_fuse_pass");
        this->predictor = paddle_infer::CreatePredictor(config);
    }

public:
    int USE_GPU;
    int EMBEDDING_SIZE;
    int SLOT_NUMBER;
    paddle::PaddlePlace place;
    std::shared_ptr<paddle_infer::Predictor> predictor;
};

template<typename TypeIn, typename TypeOut>
class PaddleInferThread {
public:
    PaddleInferThread(std::shared_ptr<PaddleInferModel> p) : piModel(p) {};

    std::vector<std::string> SplitStr(const std::string& str, const char delim)
    {
        std::stringstream is(str);
        std::vector<std::string> res;
        std::string tmp;
        while(getline(is, tmp, delim)) {
            res.push_back(tmp);
        }
        return res;
    }

    void ReadInputData()
    {
        LOG(INFO) << "reading input data" << "\n";
        if (FLAGS_trainingFile == "") {
            LOG(ERROR) << "input trainingFile is empty!" << "\n";
        }
        std::ifstream fin(FLAGS_trainingFile, std::ios_base::in);
        if (!fin.is_open()) {
            LOG(ERROR) << "input trainingFile open failed" << "\n";
        }
        BatchSample<TypeIn> samples;
        std::unordered_map<uint32_t, std::vector<int64_t>> oneSample;
        std::unordered_map<uint32_t, std::vector<uint64_t>> oneSampleFeasign;  
        int lineCnt = 0;
        long batchIdx = 0;
        std::unordered_map<std::string, int> feasignMap;
        for (std::string line; std::getline(fin, line);) {
            lineCnt++;
            std::vector<std::string> ele = SplitStr(line, ' ');
            for (size_t i = 1; i < ele.size(); i++) {
                std::vector<std::string> feature = SplitStr(ele[i], ':');
                // feasign -> embedding index
                //if (feasignMap.find(feature[0]) == feasignMap.end()) {
                //    feasignMap[feature[0]] = feasignMap.size() + 1;
                //}
                //int64_t feasign = feasignMap[feature[0]];
                uint64_t feasign = std::stoull(feature[0]);
                if (FLAGS_withCube) {
                    samples.feasignIds.insert(feasign);     
                }
                uint32_t slotId = std::stoul(feature[1]);
                oneSample[slotId].push_back(std::stoll(feature[0]));
                oneSampleFeasign[slotId].push_back(feasign);
            }
            for (auto it = slotId2name.begin(); it != slotId2name.end(); it++) { // 全量 slot
                int slotId = it->first;
                if (!oneSample[slotId].empty()) {
                    samples.features[slotId].push_back(oneSample[slotId]);
                    samples.featureCnts[slotId].push_back(oneSample[slotId].size());
                    samples.feasigns[slotId].push_back(oneSampleFeasign[slotId]);
                } else {
                    samples.features[slotId].push_back({0});
                    samples.feasigns[slotId].push_back({0});
                    samples.featureCnts[slotId].push_back(1);
                }
            }
            oneSample.clear();
            oneSampleFeasign.clear()
            if (lineCnt == FLAGS_batchSize) {
                lineCnt = 0;
                samples.batchIdx = batchIdx;
                batchIdx++;
                batchSamples.push_back(samples);
                samples.clear();
            }
        }
        int i = 0;
        int batchNum = batchSamples.size();
        while(i < FLAGS_iterationNum / FLAGS_batchSize) {
            i++;
            batchSamples.push_back(batchSamples[i % batchNum]);
        }
    }

    void GetSlotNameById()
    {
        if (inputVarNames.empty()) {
            GetInputVarNames();
        }
        for (uint i = 2; i <= piModel->SLOT_NUMBER; i++) {
            //slotId2name[std::stoul(name)] = name;
            slotId2name[i] = std::to_string(i);
        }
    }

    void CreateTensorBySlotId()
    {
        for (auto x : slotId2name) {
            uint32_t slotId = x.first;
            std::shared_ptr<paddle_infer::Tensor> lodTensor;
            lodTensor = piModel->predictor->GetInputHandle(x.second); 
            if (slotIdTensorMap.find(slotId) != slotIdTensorMap.end()) {
                return;
            }
            slotIdTensorMap[slotId] = lodTensor;
        }
    }

    paddle_infer::Tensor* GetLodTensorBySlotId(uint32_t slotId)
    {
        auto iter = slotIdTensorMap.find(slotId);
        if (iter == slotIdTensorMap.end()) {
            return nullptr;
        }
        return iter->second.get();
    }

    void CreateTensorByVarName()
    {
        if (inputVarNames.empty()) {
            GetInputVarNames();
        }
        for (auto x : inputVarNames) {
            std::shared_ptr<paddle_infer::Tensor> lodTensor;
            lodTensor = piModel->predictor->GetInputHandle(x); 
            if (varNameTensorMap.find(x) != varNameTensorMap.end()) {
                return;
            }
            varNameTensorMap[x] = lodTensor;
        }
    }

    paddle_infer::Tensor* GetLodTensorByVarName(std::string varName)
    {
        {
            auto iter = varNameTensorMap.find(varName);
            if (iter == varNameTensorMap.end()) {
                return nullptr;
            }
            return iter->second.get();
        }
    }

    void FillLodTensorWithEmbdingIdx(BatchSample<TypeIn>& batchSample)
    {
        std::vector<std::vector<size_t>> lod(1, std::vector<size_t>(FLAGS_batchSize + 1));

        for (auto it = slotId2name.begin(); it != slotId2name.end(); it++) {
            uint32_t slotId = it->first;
            paddle_infer::Tensor *lodTensor = GetLodTensorBySlotId(slotId); 
            if (lodTensor == nullptr) {
                continue;
            }
            std::vector<size_t> lod0 = {0};
            int width = 0;
            for (int sampleIdx = 0; sampleIdx < FLAGS_batchSize; ++sampleIdx) {
                int len = batchSample.featureCnts[slotId][sampleIdx];
                lod0.push_back(lod0.back() + len);
                width += batchSample.featureCnts[slotId][sampleIdx];
            }
            memcpy(lod[0].data(), lod0.data(), sizeof(size_t) * lod0.size()); // low performance
            lodTensor->SetLoD(lod);

            lodTensor->Reshape({width, 1});

            std::vector<int> v = lodTensor->shape();

            int offset = 0;
            for (int sampleIdx = 0; sampleIdx < FLAGS_batchSize; ++sampleIdx) {
                TypeIn *data_ptr = lodTensor->mutable_data<TypeIn>(piModel->place) + offset;
                memcpy(data_ptr,
                    batchSample.features[slotId][sampleIdx].data(),
                    sizeof(TypeIn) * batchSample.featureCnts[slotId][sampleIdx]);
                offset += batchSample.featureCnts[slotId][sampleIdx];
            }
        }
    }

    void QueryEmbdingVecs(std::set<uint64_t>& s, std::unordered_map<uint64_t, std::vector<float>>& queryResult)
    {
        if(s.size() == 0) {
            LOG(INFO) << "keys to query is null";
        }
	    rec::mcube::run_m(s, queryResult);
    }

    void FillLodTensorWithEmbdingVec(BatchSample<TypeIn>& batchSample, std::unordered_map<uint64_t, std::vector<float>>& queryResult)
    {
        //LOG(INFO) << "enter FillLodTensorWithEmbdingVec ...";
        queryResult[0] = std::vector<float>(piModel->EMBEDDING_SIZE, 0.0);
        std::vector<std::vector<size_t>> lod(1, std::vector<size_t>(FLAGS_batchSize + 1));
        uint feasignCnt = 0;
        uint feasignNum = batchSample.feasignIds.size();
        for (uint i = 0; i < inputVarNames.size(); i++) {
            uint32_t slotId = i + 2; 
            paddle_infer::Tensor *lodTensor = GetLodTensorByVarName(inputVarNames[i]); 
            if (lodTensor == nullptr) {
                continue;
            }
            std::vector<size_t> lod0 = {0};
            int width = 0;
            for (int sampleIdx = 0; sampleIdx < FLAGS_batchSize; ++sampleIdx) {
                int len = batchSample.featureCnts[slotId][sampleIdx];
                lod0.push_back(lod0.back() + len);
                width += len;
            }
            memcpy(lod[0].data(), lod0.data(), sizeof(size_t) * lod0.size()); // low performance
            lodTensor->SetLoD(lod);

            lodTensor->Reshape({width, piModel->EMBEDDING_SIZE});

            std::vector<int> v = lodTensor->shape();
            int offset = 0;
            for (int sampleIdx = 0; sampleIdx < FLAGS_batchSize; ++sampleIdx) {
                for (uint k = 0; k < batchSample.features[slotId][sampleIdx].size(); k++) {
                    uint64_t feasign = batchSample.feasigns[slotId][sampleIdx][k];
                    //uint64_t feasign = globalKeys[feasignCnt % feasignNum];
                    feasignCnt++;
                    TypeIn *data_ptr = lodTensor->mutable_data<TypeIn>(piModel->place) + offset;
                    memcpy(data_ptr,
                        queryResult[feasign].data(),
                        sizeof(TypeIn) * queryResult[feasign].size());
                    offset += piModel->EMBEDDING_SIZE;
                }
            }
        }
        //LOG(INFO) << "exit FillLodTensorWithEmbdingVec"; 
    }

    void GetInputVarNames() 
    {
        if (piModel == nullptr || piModel->predictor == nullptr) {
            LOG(ERROR) << "predictor is null!";
            return;
        } 
        inputVarNames = piModel->predictor->GetInputNames();
    }

    void GetOutputVarName() 
    {
        if (piModel == nullptr || piModel->predictor == nullptr) {
            LOG(ERROR) << "predictor is null!";
            return;
        } 
        outputVarNames = piModel->predictor->GetOutputNames();
    }
    void GetInferResult(BatchSample<TypeIn>& batchSample)
    {
        outputVarNames = piModel->predictor->GetOutputNames();
        for (uint32_t i = 0; i < outputVarNames.size(); i++) {
            auto outputTensor = piModel->predictor->GetOutputHandle(outputVarNames[i]);
            std::vector<int> outputShape = outputTensor->shape();
            int outNum = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int>());
            std::vector<TypeOut> out;
            out.resize(outNum);
            outputTensor->CopyToCpu(out.data());
            batchSample.outputData.push_back(out);
        }
    }
    
public:
    std::shared_ptr<PaddleInferModel> piModel;
    std::map<uint32_t, std::shared_ptr<paddle_infer::Tensor>> slotIdTensorMap;
    std::map<std::string, std::shared_ptr<paddle_infer::Tensor>> varNameTensorMap;
    std::unordered_map<uint32_t, std::string> slotId2name;
    std::vector<std::string> inputVarNames;
    std::vector<std::string> outputVarNames;
};
#endif
