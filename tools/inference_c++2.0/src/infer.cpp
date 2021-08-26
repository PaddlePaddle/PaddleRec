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

#include "../include/infer.h"

void WritePerformance(Metric& metric)
{
    LOG(INFO) << "write performance file" << "\n";
    metric.latency = metric.ms / metric.samplesCnt;
    metric.qps = 1000 * metric.samplesCnt / metric.ms;
    std::ofstream out(FLAGS_performanceFile, std::ios::app);

    if (!out.is_open()) {
        LOG(ERROR) << "performance file open failed!";
        return;
    }
    out << "threadNum: " << FLAGS_threadNum <<
        " " << "batchSize: " << FLAGS_batchSize << " totalSampleCnts: " << metric.samplesCnt << "\n";
    out << "latency per sample: " << metric.latency << " ms" <<
        " QPS: " << metric.qps << "\n";
        
    float predictorQps = 1000 * metric.samplesCnt / metric.predictorTimeCost;
    float predictorLatency = metric.predictorTimeCost / metric.samplesCnt;
    out << "paddle predictor latency: " << predictorLatency << " ms" << "\n" 
        "paddle predictor qps: " << predictorQps << "\n";
float cubeQps = 1000 * metric.samplesCnt / metric.cubeTimeCost;
    float cubeLatency = metric.cubeTimeCost / metric.samplesCnt;
    out << "cube query latency: " << cubeLatency << " ms" << "\n"
        "cube query qps: " << cubeQps << "\n";
    out << "+++++++++++++++++++++++++\n";
    out.close();
    return;
};
