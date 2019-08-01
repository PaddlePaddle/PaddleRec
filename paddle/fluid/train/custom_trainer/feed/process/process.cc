#include "paddle/fluid/train/custom_trainer/feed/process/process.h"
#include "paddle/fluid/train/custom_trainer/feed/process/init_env_process.h"
#include "paddle/fluid/train/custom_trainer/feed/process/learner_process.h"

namespace paddle {
namespace custom_trainer {
namespace feed {
REGISTER_CLASS(Process, InitEnvProcess);
REGISTER_CLASS(Process, LearnerProcess);
int Process::run() {
    return 0;
}


}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
