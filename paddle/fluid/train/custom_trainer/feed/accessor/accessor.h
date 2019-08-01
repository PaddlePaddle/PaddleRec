#pragma once
#include "paddle/fluid/train/custom_trainer/feed/common/registerer.h"
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

class Accessor {
public:
    Accessor() {}
    virtual ~Accessor() {}
    virtual int initialize(YAML::Node config,
        std::shared_ptr<TrainerContext> context_ptr) = 0;
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
