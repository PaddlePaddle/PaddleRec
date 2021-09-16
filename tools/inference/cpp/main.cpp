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

#include "include/debug.h"
#include "include/infer.h"
#include "include/cube_cli.h"
#include "include/single.h"
#include <functional>
#include <sched.h>
#include <pthread.h>

std::vector<uint64_t> globalKeys;
DEFINE_bool(testPredictor, true, "test Predictor");
DEFINE_bool(testCubeCost, true, "test cube query cost");
DEFINE_bool(debug, true, "debug switch");
DEFINE_bool(withCube, true, "with cube");
DEFINE_int32(threadNum, 2, "thread num");
DEFINE_int32(batchSize, 3, "batch size");
DEFINE_int32(iterationNum, 100, "iteration num");
DEFINE_string(performanceFile, "../performance.txt", "performanceFile");
DEFINE_string(trainingFile, "../../data/out_test.1", "input data file");
DEFINE_string(modelFile, "../../data/rec_inference/rec_inference.pdmodel", "model file");
DEFINE_string(paramFile, "../../data/rec_inference/rec_inference.pdiparams", "param file");
DEFINE_string(predictorLog, "../predictor.log", "predictor log");
DEFINE_string(stdLog, "../std.log", "standard log");
DEFINE_string(varsName, "../all_vars.txt", "all vars name");
DEFINE_string(keys, "../keys", "keys to seek");
DEFINE_uint64(cube_batch_size, 1, "cube batch size");

static int currResIdx = 0;
static bool update = false;

SharedQueue<BatchSample<dataType>> batchSamples;

template<typename TypeIn, typename TypeOut>
class ThreadParams {
public:
    ThreadParams (int tid) : tid(tid) {};
    void Init()
    {
        piModel = std::make_shared<PaddleInferModel>();
        tr = std::make_shared<PaddleInferThread<TypeIn, TypeOut>>(piModel);
        pd = std::make_shared<PaddleInferDebug<TypeIn, TypeOut>>(piModel, tr);
    }

public:
    int tid;
    std::shared_ptr<PaddleInferModel> piModel;
    std::shared_ptr<PaddleInferThread<TypeIn, TypeOut>> tr;
    std::shared_ptr<PaddleInferDebug<TypeIn, TypeOut>> pd;
};

std::shared_ptr<ThreadParams<dataType, float>> resBuf[2][128] = {{}};

void CreateInferThreads(int threadNum)
{
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < threadNum; j++) {
            std::shared_ptr<ThreadParams<dataType, float>> sp = std::make_shared<ThreadParams<dataType, float>>(j + 1);
            sp->Init();
            resBuf[i][j] = sp;
        }
    }
}

#ifdef USE_LOCK
void InferMain(int tid, std::vector<Metric>& metrics)
{
    std::chrono::duration<float> tPredictor(0.0), tCube(0.0);
    std::chrono::time_point<std::chrono::system_clock> tPredictor1, tPredictor2, tCube1, tCube2;

    LOG(INFO) << "thread " << tid << " start";
    LOG(INFO) << "Thread #" << tid << ":on CPU " << sched_getcpu();
    std::shared_ptr<ThreadParams<dataType, float>> tp = resBuf[currResIdx][tid];
    if (tp == nullptr) {
        LOG(ERROR) << "ThreadParams object is null" << "\n";
    }
    tp->piModel->ReadConfig();
    tp->piModel->LoadModel();
    if (FLAGS_debug) {
        tp->pd->TestPredictor();
    }
    tp->tr->GetSlotNameById();
    if (!FLAGS_withCube) {
        tp->tr->CreateTensorBySlotId();
    } else {
        tp->tr->CreateTensorByVarName();
    }
    while(!batchSamples.empty()) {
        if (!FLAGS_withCube) {
	        BatchSample<dataType> batchSample = batchSamples.pop();
           if (FLAGS_debug) {
   	         LOG(INFO) << "thread id: " << tid << " batch idx: " << batchSample.batchIdx;
       	    }
            tp->tr->FillLodTensorWithEmbdingIdx(batchSample);
            if (FLAGS_testPredictor) {
                tPredictor1 = std::chrono::high_resolution_clock::now();
            }
            tp->tr->piModel->predictor->Run();
            if (FLAGS_testPredictor) {
                tPredictor2 = std::chrono::high_resolution_clock::now();
                tPredictor += (tPredictor2 - tPredictor1);
            }
            tp->tr->GetInferResult(batchSample);
            if (FLAGS_debug) {
                tp->pd->PrintInferResult(tid, batchSample);
                tp->pd->WriteLayersOutput(tid, batchSample.batchIdx);
                tp->pd->WriteStdOutput(tid, batchSample);
            }
        } else {
            std::vector<BatchSample<dataType>> batches(1024);
            std::set<uint64_t> s;
            std::unordered_map<uint64_t, std::vector<float>> queryResult;
            int cubeBatchCnt = 0;
            while(true) {
                BatchSample<dataType> batchSample = batchSamples.pop();
                if (FLAGS_debug) {
                    LOG(INFO) << "thread id: " << tid << " batch idx: " << batchSample.batchIdx;
                }
                s.insert(batchSample.feasignIds.begin(), batchSample.feasignIds.end());
                batches[cubeBatchCnt] = std::move(batchSample);
                cubeBatchCnt++;
                if (cubeBatchCnt * FLAGS_batchSize >= FLAGS_cube_batch_size) {
                    break;
                }
                if (batchSamples.empty() || cubeBatchCnt >= batches.size()) {
                    break;
                }
            }
	        //LOG(INFO) << "query cube by batch: " << cubeBatchCnt;
            if (FLAGS_testCubeCost) {
                tCube1 = std::chrono::high_resolution_clock::now();	
            }
            tp->tr->QueryEmbdingVecs(s, queryResult);
           	if (FLAGS_testCubeCost) {
                tCube2 = std::chrono::high_resolution_clock::now();
                tCube += (tCube2 - tCube1);
            }
	        for (uint i = 0; i < cubeBatchCnt; i++) {
                //LOG(INFO) << "procssing batch: " << i;
                tp->tr->FillLodTensorWithEmbdingVec(batches[i], queryResult);
                if (FLAGS_debug) {
                    tp->pd->PrintLodTensorByBatchIdx(batches[i].batchIdx);
                }
                if (FLAGS_testPredictor) {
                    tPredictor1 = std::chrono::high_resolution_clock::now();
                }
                tp->tr->piModel->predictor->Run();
                if (FLAGS_testPredictor) {
                    tPredictor2 = std::chrono::high_resolution_clock::now();
                    tPredictor += (tPredictor2 - tPredictor1);
                }
                tp->tr->GetInferResult(batches[i]);
                if (FLAGS_debug) {
                    tp->pd->PrintInferResult(tid, batches[i]);
                    tp->pd->WriteLayersOutput(tid, batches[i].batchIdx);
                    tp->pd->WriteStdOutput(tid, batches[i]);
                }
            }
        }
    }
    Metric metric;
    metric.predictorTimeCost = tPredictor.count() * 1000.0f;
    metric.cubeTimeCost = tCube.count() * 1000.0f;
    metrics[tid] = std::move(metric);
    LOG(INFO) << "thread " << tid << " over!";
}
#else
void InferMain(int tid, std::vector<Metric>& metrics)
{
    std::chrono::duration<float> tPredictor(0.0), tCube(0.0);
    std::chrono::time_point<std::chrono::system_clock> tPredictor1, tPredictor2, tCube1, tCube2;

    LOG(INFO) << "thread " << tid << " start";
    LOG(INFO) << "Thread #" << tid << ":on CPU " << sched_getcpu();
    std::shared_ptr<ThreadParams<dataType, float>> tp = resBuf[currResIdx][tid];
    if (tp == nullptr) {
        LOG(ERROR) << "ThreadParams object is null" << "\n";
    }
    tp->piModel->ReadConfig();
    tp->piModel->LoadModel();
    if (FLAGS_debug) {
        tp->pd->TestPredictor();
    }
    tp->tr->GetSlotNameById();
    if (!FLAGS_withCube) {
        tp->tr->CreateTensorBySlotId();
    } else {
        tp->tr->CreateTensorByVarName();
    }
    int batchNum = batchSamples.size();
    int k = 0;
    int sz = batchNum / FLAGS_threadNum;
    while(k < sz) {
        if (!FLAGS_withCube) {
            BatchSample<dataType> batchSample = batchSamples[k + tid * sz];
            k++;
            if (FLAGS_debug) {
   	            LOG(INFO) << "thread id: " << tid << " batch idx: " << batchSample.batchIdx;
       	    }
            tp->tr->FillLodTensorWithEmbdingIdx(batchSample);
            if (FLAGS_testPredictor) {
                tPredictor1 = std::chrono::high_resolution_clock::now();
            }
            tp->tr->piModel->predictor->Run();
            if (FLAGS_testPredictor) {
                tPredictor2 = std::chrono::high_resolution_clock::now();
                tPredictor += (tPredictor2 - tPredictor1);
            }
            tp->tr->GetInferResult(batchSample);
            if (FLAGS_debug) {
                tp->pd->PrintInferResult(tid, batchSample);
                tp->pd->WriteLayersOutput(tid, batchSample.batchIdx);
                tp->pd->WriteStdOutput(tid, batchSample);
            }
        } else {
            std::vector<BatchSample<dataType>> batches(1024);
            std::set<uint64_t> s;
            std::unordered_map<uint64_t, std::vector<float>> queryResult;
            int cubeBatchCnt = 0;
            while(true) {
                BatchSample<dataType> batchSample = batchSamples[k + tid * sz];
                k++;
                if (FLAGS_debug) {
                    LOG(INFO) << "thread id: " << tid << " batch idx: " << batchSample.batchIdx;
                }
                s.insert(batchSample.feasignIds.begin(), batchSample.feasignIds.end());
                batches[cubeBatchCnt] = std::move(batchSample);
                cubeBatchCnt++;
                if (cubeBatchCnt * FLAGS_batchSize >= FLAGS_cube_batch_size) {
                    break;
                }
                if (k >= sz || cubeBatchCnt >= batches.size()) {
                    break;
                }
            }
            if (FLAGS_testCubeCost) {
                tCube1 = std::chrono::high_resolution_clock::now();
            }
            tp->tr->QueryEmbdingVecs(s, queryResult);
            if (FLAGS_testCubeCost) {
                tCube2 = std::chrono::high_resolution_clock::now();
                tCube += (tCube2 - tCube1);
            }
            for (int i = 0; i < cubeBatchCnt; i++) {
                //LOG(INFO) << "procssing batch: " << i;
                tp->tr->FillLodTensorWithEmbdingVec(batches[i], queryResult);
                if (FLAGS_debug) {
                    tp->pd->PrintLodTensorByBatchIdx(batches[i].batchIdx);
                }
                if (FLAGS_testPredictor) {
                    tPredictor1 = std::chrono::high_resolution_clock::now();
                }
                tp->tr->piModel->predictor->Run();
                if (FLAGS_testPredictor) {
                    tPredictor2 = std::chrono::high_resolution_clock::now();
                    tPredictor += (tPredictor2 - tPredictor1);
                }
                tp->tr->GetInferResult(batches[i]);
                if (FLAGS_debug) {
                    tp->pd->PrintInferResult(tid, batches[i]);
                    tp->pd->WriteLayersOutput(tid, batches[i].batchIdx);
                    tp->pd->WriteStdOutput(tid, batches[i]);
                }
            }
        }
    }
    Metric metric;
    metric.predictorTimeCost = tPredictor.count() * 1000.0f;
    metric.cubeTimeCost = tCube.count() * 1000.0f;
    metrics[tid] = std::move(metric);
    LOG(INFO) << "thread " << tid << " over!";
}
#endif

void Listen()
{
    int timeCost = 0;
    while(true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        timeCost += 10;
        LOG(INFO) << " listen thread id: " << 0 << " time passed: " << timeCost << std::endl;
        if (update) {
            currResIdx = 1 - currResIdx;
        }
    }
}

int main(int argc, char *argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    std::ifstream keyFile(FLAGS_keys.c_str());
    if (!keyFile.is_open()) {
        LOG(ERROR) << "cube key file open failed!";
    }
    std::string line;
    while (getline(keyFile, line)) {
        globalKeys.push_back(std::stoul(line));
    }
    keyFile.close();

    WriteCubeQryResult();
    // 1. 启动监听线程
    //std::thread listenThread(Listen);
    //listenThread.detach();

    // 2. 从文件读 input 数据，写入全局队列
    std::shared_ptr<PaddleInferModel> model = std::make_shared<PaddleInferModel>();
    std::shared_ptr<PaddleInferThread<dataType, float>> reader = std::make_shared<PaddleInferThread<dataType, float>>(model);
    model->ReadConfig();
    model->LoadModel();
    reader->GetSlotNameById();
    reader->ReadInputData();

    // 3.1 创建推理线程
    CreateInferThreads(FLAGS_threadNum);
    // 3.2 启动推理线程
    std::vector<Metric> metrics(FLAGS_threadNum);
    int batchNum = batchSamples.size();
    std::vector<std::thread> threads;
    std::chrono::duration<float> duration(0.0);
    std::chrono::time_point<std::chrono::system_clock> t1, t2;
    t1 = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < FLAGS_threadNum; j++) {
        std::thread inferThread(InferMain, j, std::ref(metrics));
        threads.push_back(std::move(inferThread));
    }
    /*
    std::vector<std::thread> threads;
    for (int j = 0; j < FLAGS_threadNum; j++) {
        std::thread inferThread(InferMain, j);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(j, &cpuset);
        pthread_setaffinity_np(inferThread.native_handle(), sizeof(cpu_set_t), &cpuset);
        threads.push_back(std::move(inferThread));
    }
    */
    for (int j = 0; j < FLAGS_threadNum; j++) {
        threads[j].join();
    }
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    float ms = duration.count() * 1000.0f;
    LOG(INFO) << "e2e time cost: " << ms << "\n";
    Metric globalMetric;
    globalMetric.ms = ms;
    globalMetric.samplesCnt = batchNum * FLAGS_batchSize;
    float tp = 0.0;
    float tc = 0.0;
    for (auto metric : metrics) {
        tp += metric.predictorTimeCost;
        tc += metric.cubeTimeCost;
    }
    globalMetric.predictorTimeCost = tp / metrics.size();
    globalMetric.cubeTimeCost = tc / metrics.size();
    LOG(INFO) << "predictor time cost: " << globalMetric.predictorTimeCost << "\n";
    LOG(INFO) << "cube query time cost: " << globalMetric.cubeTimeCost << "\n";
    WritePerformance(globalMetric);
    return 0;
}
