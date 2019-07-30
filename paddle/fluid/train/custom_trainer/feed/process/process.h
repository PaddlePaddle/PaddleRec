#pragma once
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class Process {
public:
    Process() {}
    virtual ~Process() {}
    virtual int initialize(std::shared_ptr<TrainerContext> context_ptr) = 0;
    virtual int run();
};
REGISTER_REGISTERER(Process);

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
