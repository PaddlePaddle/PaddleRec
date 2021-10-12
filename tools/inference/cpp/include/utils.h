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

#ifndef BAIDU_PADDLEPADDLE_PADDLEREC_UTILS_H
#define BAIDU_PADDLEPADDLE_PADDLEREC_UTILS_H

#include <iostream>
#include <fstream>
#include <queue>
#include <mutex>
#include <set>
#include <condition_variable>
#include <thread>
#include <string>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <ostream>
#include <unordered_map>
#include <chrono>            // 适用于多种平台
#include "gflags/gflags.h"
#include "brpc/server.h"
//#include "glog/logging.h"

template <typename T>
class BatchSample
{
public:
    void clear()
    {
        slotIds.clear();
        feasignIds.clear();
        features.erase(features.begin(), features.end());
        feasigns.erase(feasigns.begin(), feasigns.end());
        featureCnts.erase(featureCnts.begin(), featureCnts.end());
    }

    int size()
    {
        if (features.size() == featureCnts.size()) {
            return features.size();
        } else {
            return 0;
        }
    }

public:
    long batchIdx;
    std::vector<uint32_t> slotIds;
    std::set<uint64_t> feasignIds;
    std::unordered_map<uint32_t, std::vector<std::vector<int64_t>>> features; // key1: slotId
    std::unordered_map<uint32_t, std::vector<std::vector<uint64_t>>> feasigns; // key1: slotId
    std::unordered_map<uint32_t, std::vector<uint32_t>> featureCnts;

    std::unordered_map<uint64_t, std::vector<float>> queryResult; // key: feasign, value: embdding vector

    std::vector<std::vector<float>> outputData;
};

struct Metric
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<float> duration;
    float ms = 0.0;
    float latency;
    float qps;
    long samplesCnt = 0;
    float predictorTimeCost = 0.0;
    float cubeTimeCost = 0.0;
};

void stringTokenize(const char *str, const char *tokens, std::vector<std::string> &ret);

// see: https://stackoverflow.com/questions/36762248/why-is-stdqueue-not-thread-safe

template <typename T>
class SharedQueue
{
public:
    SharedQueue();

    ~SharedQueue();

    T &front();

    T pop();

    void pop_front();

    void push_back(const T &item);

    void push_back(T &&item);

    void shut_down();

    int size();

    bool empty();

    bool is_shutdown();

public:
    T operator[](int k)
    {
        return queue_[k];
    }

private:
    std::deque<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool shutdown = false;
};

template <typename T>
SharedQueue<T>::SharedQueue() {}

template <typename T>
SharedQueue<T>::~SharedQueue() {}

template <typename T>
bool SharedQueue<T>::is_shutdown()
{
    return this->shutdown;
}

template <typename T>
void SharedQueue<T>::shut_down()
{
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.clear();
    this->shutdown = true;
}

template <typename T>
T SharedQueue<T>::pop()
{
    std::unique_lock<std::mutex> mlock(mutex_);
    /*
    while (queue_.empty())
    {
        cond_.wait(mlock);
    }
    */
    cond_.wait(mlock, [this](){ return !this->queue_.empty();});
    T rc(std::move(queue_.front()));
    queue_.pop_front();
    return rc;
}

template <typename T>
T &SharedQueue<T>::front()
{
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty())
    {
        cond_.wait(mlock);
    }
    return queue_.front();
}

template <typename T>
void SharedQueue<T>::pop_front()
{
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty())
    {
        cond_.wait(mlock);
    }
    queue_.pop_front();
}

template <typename T>
void SharedQueue<T>::push_back(const T &item)
{
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push_back(item);
    mlock.unlock();     // unlock before notificiation to minimize mutex con
    cond_.notify_one(); // notify one waiting thread
}

template <typename T>
void SharedQueue<T>::push_back(T &&item)
{
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push_back(std::move(item));
    mlock.unlock();     // unlock before notificiation to minimize mutex con
    cond_.notify_one(); // notify one waiting thread
}

template <typename T>
int SharedQueue<T>::size()
{
    std::unique_lock<std::mutex> mlock(mutex_);
    int size = queue_.size();
    mlock.unlock();
    return size;
}

template <typename T>
bool SharedQueue<T>::empty()
{
    std::unique_lock<std::mutex> mlock(mutex_);
    bool is_empty = queue_.empty();
    mlock.unlock();
    return is_empty;
}
#endif
