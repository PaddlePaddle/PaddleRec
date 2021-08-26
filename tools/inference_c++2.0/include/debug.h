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

#ifndef BAIDU_PADDLEPADDLE_PADDLERINFER_DEBUG_H
#define BAIDU_PADDLEPADDLE_PADDLERINFER_DEBUG_H

#include "infer.h"
#include <iomanip>

template<typename TypeIn, typename TypeOut>
class PaddleInferDebug {
public:
    PaddleInferDebug(std::shared_ptr<PaddleInferModel> p1, std::shared_ptr<PaddleInferThread<TypeIn, TypeOut>> p2)
    {
        piModel = p1;
        piThread = p2;
    }

    void PrintSlotIdPermOneSample(const std::vector<uint32_t>& slotIdsPerSample) 
    {
        std::cout << "SlotId perm in one Sample: ";
        for_each(slotIdsPerSample.begin(), slotIdsPerSample.end(), [](int ele) {std::cout << ele << ", ";});
        std::cout << "\n";
    }

    void TestPredictor()
    {
        std::ofstream fout(FLAGS_predictorLog, std::ios::app);
        fout << "++++++++++ test predictor ++++++++++" << "\n";
        piThread->GetInputVarNames();
        fout << "inputVarNames: ";
        for_each(piThread->inputVarNames.begin(), piThread->inputVarNames.end(), [&](const std::string& str){fout << str << " ";});
        fout << "\n";
        piThread->GetOutputVarName();
        fout << "outputVarNames:";
        for_each(piThread->outputVarNames.begin(), piThread->outputVarNames.end(), [&](const std::string& str){fout << str << " ";});
        fout << "\n";
        if (piThread->inputVarNames.empty() || piThread->outputVarNames.empty()) {
            fout << "Predictor is not valid!" << "\n";
        } else {
            fout << "Predictor is OK!" << "\n";
        }
        fout.close();
        return;
    }
    
    void PrintLodTensorByBatchIdx(uint32_t batchIdx)
    {
        std::ofstream fout(FLAGS_predictorLog, std::ios::app);
        fout << "++++++++++ print lod tensor ++++++++++" << "\n";
        fout << "batchIdx: " << batchIdx << "\n";
        for (auto str : piThread->inputVarNames) {
            fout << "var name: " << str << " ";
            auto lodTensor = piModel->predictor->GetInputHandle(str);
            std::vector<int> shape = lodTensor->shape();
            std::vector<TypeIn> data;
            int dim = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            data.resize(dim);

            int num;
            paddle_infer::PlaceType tp;
            fout << "tensor_shape: { ";
            for_each(shape.begin(), shape.end(), [&](int ele){fout << ele << " ";}); 
            fout << "}" << ", ";
            fout << "data_size: " << data.size() << ", " << "tensor data from handle: ";
            for (int i = 0; i < dim; ++i) {
                fout << lodTensor->template data<TypeIn>(&tp, &num)[i] << " ";
            }
            /*
            auto it = piThread->slotIdTensorMap.find(slotId);
            fout << "tensor data from map: ";
            for (int i = 0; i < dim; ++i) {
                fout << it->second->template data<TypeIn>(&tp, &num)[i] << " ";
            }
            */
            fout << "\n";
        }
        fout.close();
    }

    void PrintInferResult(int tid, BatchSample<TypeIn>& batchSample) 
    {
        std::ofstream fout;
        fout << "++++++++++ print infer result ++++++++++" << "\n";
        if (batchSample.outputData.empty()) {
            return;
        }
        fout << "thread id: " << tid << " batchIdx: " << batchSample.batchIdx << " infer result: "; 
        fout.close();
        //std::cout.precision(10);
        for (uint32_t i = 0; i < piThread->outputVarNames.size(); i++) {
            fout.open(FLAGS_predictorLog, std::ios::app);
            fout << "output var name: " << piThread->outputVarNames[i] << " ";
            fout.close();
            for (auto k : batchSample.outputData[i]) {
                fout.open(FLAGS_predictorLog, std::ios::app);
                fout << std::setw(10) << std::setfill('0') << std::setiosflags(std::ios::fixed) << std::setprecision(10) << k << " ";
                fout.close();
            }
            fout.open(FLAGS_predictorLog, std::ios::app);
            fout << "\n";
            fout.close();
        }
    }

    void WriteLayersOutput(int threadId, int batchIdx)
    {
        std::ifstream fin(FLAGS_varsName, std::ios_base::in);
        if (!fin.is_open()) {
            LOG(ERROR) << "input vars_name file open failed!";
        } 
        std::set<std::string> varNames;
        std::string line;
        while(std::getline(fin, line)) {
            varNames.insert(line);
        }
        std::ofstream fout(FLAGS_predictorLog, std::ios::app);
        fout << "++++++++++ print layers out ++++++++++" << "\n";
        fout << "threadId: " << threadId << " batchIdx: " << batchIdx << " samples: " << batchIdx * FLAGS_batchSize << " -- " << ((batchIdx + 1) * FLAGS_batchSize - 1) << "\n";
        for (auto str : varNames) {
            auto tensor = piModel->predictor->GetOutputHandle(str);
            std::vector<int> shape = tensor->shape();
            std::vector<float> data;
            long dim = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            data.resize(dim);
            int num;
            paddle_infer::PlaceType tp;
            fout << "VarName: " << str << " shape: [ ";
            for (uint i = 0; i < shape.size(); i++) {
                fout << shape[i] << " ";
            }
            fout << "]\n";
            fout << "data: ";
            auto dt = tensor->type();
            if (dt == paddle_infer::DataType::FLOAT32) {
                for (int i = 0; i < dim; ++i) {
                    fout << tensor->template data<float>(&tp, &num)[i] << " ";
                }
            } else if (dt == paddle_infer::DataType::INT64) {
                for (int i = 0; i < dim; ++i) {
                    fout << tensor->template data<int64_t>(&tp, &num)[i] << " ";
                }
            } else if (dt == paddle_infer::DataType::INT32) {
                for (int i = 0; i < dim; ++i) {
                    fout << tensor->template data<int32_t>(&tp, &num)[i] << " ";
                }
            } else if (dt == paddle_infer::DataType::INT8) {
                for (int i = 0; i < dim; ++i) {
                    fout << tensor->template data<int8_t>(&tp, &num)[i] << " ";
                }
            }
            fout << "\n";
        }
        fin.close();
        fout.close();
    }

    void WriteStdOutput(int threadId, BatchSample<TypeIn>& batchSample)
    {
        std::ifstream fin(FLAGS_varsName, std::ios_base::in);
        if (!fin.is_open()) {
            LOG(ERROR) << "input vars_name file open failed!";
        } 
        std::set<std::string> varNames;
        std::string line;
        while(std::getline(fin, line)) {
            varNames.insert(line);
        }
        long batchIdx = batchSample.batchIdx;
        std::ofstream fout(FLAGS_stdLog, std::ios::app);
        for (auto str : varNames) {
            fout << "[threadId:" << threadId << "]";
            fout << "[batchIdx:" << batchIdx << "]";
            fout << "[samples:" << batchIdx * FLAGS_batchSize << "--" << ((batchIdx + 1) * FLAGS_batchSize - 1) << "]";
            auto tensor = piModel->predictor->GetOutputHandle(str);
            std::vector<int> shape = tensor->shape();
            std::vector<float> data;
            long dim = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            data.resize(dim);
            int num;
            paddle_infer::PlaceType tp;
            fout << "[Tensor]" << "[name:" << str << "]" << "[shape:";
            for (uint i = 0; i < shape.size(); i++) {
                fout << shape[i];
                if (i != (shape.size() - 1)) {
                    fout << ",";
                }
            }
            fout << "]";
            if (!FLAGS_withCube) {
                if (std::find(piThread->inputVarNames.begin(), piThread->inputVarNames.end(), str) != piThread->inputVarNames.end()) {
                    uint32_t slotId = std::stoul(str);
                    fout << "[lod:|0|0,";
                    int tmp = 0;
                    for (uint i = 0; i < batchSample.featureCnts[slotId].size(); i++) {
                        int len = tmp + batchSample.featureCnts[slotId][i];
                        tmp = len;
                        fout << len; 
                        if (i != (batchSample.featureCnts[slotId].size() - 1)) {
                        fout << ",";
                        }
                    }
                } else {
                    fout << "[lod:|0|0";
                }
                fout << "]";
            }
            auto dt = tensor->type();
            if (dt == paddle_infer::DataType::FLOAT32) {
                fout << "[dtype:float32]";
                fout << "[size:" << dim << "]";
                fout << "[offset:" << 0 << "]";
                fout << "[buf size:" << dim << "]";
                fout << "[buf capacity:" << dim << "]";
                fout << "[data:";
                for (int i = 0; i < dim; ++i) {
                    fout << tensor->template data<float>(&tp, &num)[i];
                    if (i != (dim - 1)) {
                        fout << ",";
                    }
                }
                fout << "]";
            } else if (dt == paddle_infer::DataType::INT64) {
                fout << "[dtype:int64]";
                fout << "[size:" << dim << "]";
                fout << "[offset:" << 0 << "]";
                fout << "[buf size:" << dim << "]";
                fout << "[buf capacity:" << dim << "]";
                fout << "[data:";
                for (int i = 0; i < dim; ++i) {
                    fout << tensor->template data<int64_t>(&tp, &num)[i];
                    if (i != (dim - 1)) {
                        fout << ",";
                    }
                }
                fout << "]";
            } else if (dt == paddle_infer::DataType::INT32) {
                fout << "[dtype:int32]";
                fout << "[size:" << dim << "]";
                fout << "[offset:" << 0 << "]";
                fout << "[buf size:" << dim << "]";
                fout << "[buf capacity:" << dim << "]";
                fout << "[data:";
                for (int i = 0; i < dim; ++i) {
                    fout << tensor->template data<int32_t>(&tp, &num)[i];
                    if (i != (dim - 1)) {
                        fout << ",";
                    }
                }
                fout << "]";
            } else if (dt == paddle_infer::DataType::INT8) {
                fout << "[dtype:int8]";
                fout << "[size:" << dim << "]";
                fout << "[offset:" << 0 << "]";
                fout << "[buf size:" << dim << "]";
                fout << "[buf capacity:" << dim << "]";
                fout << "[data:";
                for (int i = 0; i < dim; ++i) {
                    fout << tensor->template data<int8_t>(&tp, &num)[i];
                    if (i != (dim - 1)) {
                        fout << ",";
                    }
                }
                fout << "]";
            }
            fout << "\n";
        }
        fin.close();
        fout.close();
    }

public:
    std::shared_ptr<PaddleInferModel> piModel;
    std::shared_ptr<PaddleInferThread<TypeIn, TypeOut>> piThread;
};

void WriteCubeQryResult();
#endif
