#pragma once

#ifndef BUTIL_LOGGING_H_
#define BUTIL_LOGGING_H_
#endif

#include <tuple>
#include <future>
#include <functional>
#include <forward_list>
#include "glog/logging.h"
#include "google/protobuf/stubs/callback.h"
#include "bthread/bthread.h"
#include "bthread/mutex.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

void* execute_bthread_task(void* args);

class BthreadTaskRunner {
public: 
    static BthreadTaskRunner& instance() {
        static BthreadTaskRunner runner;
        return runner;
    }
    
    template <class Callable, class... Args>
    int add_task(Callable &&func, Args &&... args) {
        bthread_t th;
        auto* task = new std::packaged_task<void()>(
            std::bind(std::forward<Callable>(func), std::forward<Args>(args)...));
        auto* param = new ::std::tuple<std::packaged_task<void()>*, google::protobuf::Closure*>(
            ::std::move(task), NULL);
        if (0 != bthread_start_background(&th, NULL, execute_bthread_task, param)) {
            delete task;
            delete param;
            return -1;
        }
        return 0;
    }
private:
    BthreadTaskRunner() {}
    ~BthreadTaskRunner() {}
};

} // namespace feed
} // namespace custom_trainer
} // namespace paddle
