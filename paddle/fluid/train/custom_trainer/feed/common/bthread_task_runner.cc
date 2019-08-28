#include "paddle/fluid/train/custom_trainer/feed/common/bthread_task_runner.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

void* execute_bthread_task(void* args) {
    auto* param = reinterpret_cast<::std::tuple<std::packaged_task<void()>*, google::protobuf::Closure*>*>(args);
    auto* task = ::std::get<0>(*param);
    auto* closure = ::std::get<1>(*param);
    (*task)();
    if (closure != NULL) {
        closure->Run();
    }
    delete task;
    delete param;
    return NULL;
}

} // namespace feed
} // namespace custom_trainer
} // namespace paddle
