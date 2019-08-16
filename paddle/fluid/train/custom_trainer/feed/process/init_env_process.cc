/*
 *Author: xiexionghang
 *用于训练环境的整体配置读取、环境初始化工作
 */
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/train/custom_trainer/feed/io/file_system.h"
#include "paddle/fluid/train/custom_trainer/feed/dataset/dataset.h"
#include "paddle/fluid/train/custom_trainer/feed/accessor/epoch_accessor.h"
#include "paddle/fluid/train/custom_trainer/feed/process/init_env_process.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int InitEnvProcess::initialize(std::shared_ptr<TrainerContext> context_ptr) {
    Process::initialize(context_ptr);
    paddle::framework::InitDevices(false);
    context_ptr->cpu_place = paddle::platform::CPUPlace();
    
    YAML::Node config = _context_ptr->trainer_config;

    //file_system
    context_ptr->file_system.reset(CREATE_INSTANCE(FileSystem, "AutoFileSystem"));
    if (context_ptr->file_system->initialize(config["io"], context_ptr) != 0) {
        return -1;
    }

    //epoch
    std::string epoch_class = config["epoch"]["epoch_class"].as<std::string>();
    context_ptr->epoch_accessor.reset(CREATE_INSTANCE(EpochAccessor, epoch_class));
    if (context_ptr->epoch_accessor->initialize(config["epoch"], context_ptr) != 0) {
        return -1;
    }

    //Dataset
    context_ptr->dataset.reset(new Dataset());
    if (context_ptr->dataset->initialize(config["dataset"], context_ptr) != 0) {
        return -1;
    }
    
    VLOG(3) << "Env initialize success"; 
    return 0;
}

int InitEnvProcess::run() {
    auto* epoch_accessor = _context_ptr->epoch_accessor.get();
    VLOG(3) << "Trainer Resume From epoch:" << epoch_accessor->current_epoch_id();
    auto next_epoch_id = epoch_accessor->next_epoch_id(epoch_accessor->current_epoch_id());
    _context_ptr->dataset->pre_detect_data(next_epoch_id);

    if (epoch_accessor->checkpoint_path().size() > 0) {
        //Load Model
    } else {
        //Random Init Model
    }
    //context_ptr->pslib_client()->load_model();
    VLOG(3) << "Psserver Load Model Success";
    return 0;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
