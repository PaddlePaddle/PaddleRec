#include <time.h>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "paddle/fluid/train/custom_trainer/feed/trainer_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/train/custom_trainer/feed/process/process.h"
#include "paddle/fluid/train/custom_trainer/feed/process/init_env_process.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/pybind/pybind.h"

using namespace paddle::custom_trainer::feed;

DEFINE_string(feed_trainer_conf_path, "./conf/trainer.yaml", "path of trainer conf");

int main(int argc, char* argv[]) {
    //gflags
    google::ParseCommandLineFlags(&argc, &argv, true);
    std::string gflag_conf = "./conf/gflags.conf";
    google::SetCommandLineOption("flagfile", gflag_conf.c_str()); 

    //load trainer config
    auto trainer_context_ptr = std::make_shared<TrainerContext>();
    trainer_context_ptr->trainer_config = YAML::LoadFile(FLAGS_feed_trainer_conf_path);    

    //environment
    auto& config = trainer_context_ptr->trainer_config;
    std::string env_class = config["environment"]["environment_class"].as<std::string>();
    trainer_context_ptr->environment.reset(CREATE_INSTANCE(RuntimeEnvironment, env_class));
    if (trainer_context_ptr->environment->initialize(config["environment"]) != 0) {
        return -1;
    }
    EnvironmentRole role;
    auto* environment = trainer_context_ptr->environment.get();
    environment->wireup();
    if (environment->rank_id(EnvironmentRole::ALL) % 2 == 0) {
        role = EnvironmentRole::WORKER;
    } else {
        role = EnvironmentRole::PSERVER;
    } 
    environment->set_role(role);
    trainer_context_ptr->pslib.reset(new PSlib());
    std::string ps_config = config["environment"]["ps"].as<std::string>();
    trainer_context_ptr->pslib->initialize(ps_config, environment, role);
    //VLOG(3) << "Node Start With Role:" << role;    
     
    std::vector<std::string> process_name_list = {
        "InitEnvProcess",
        "LearnerProcess"
    };
    switch (role) {
    case EnvironmentRole::WORKER:
        for (const auto& process_name : process_name_list) {
            Process* process = CREATE_INSTANCE(Process, process_name);
            if (process == NULL) {
                VLOG(1) << "Process:" << process_name << " does not exist"; 
                return -1;
            }
            if (process->initialize(trainer_context_ptr) != 0) {
                VLOG(1) << "Process:" << process_name << " initialize failed"; 
                return -1;
            }
            trainer_context_ptr->process_list.push_back(std::shared_ptr<Process>(process));
        } 
        for (auto& process : trainer_context_ptr->process_list) {
            process->run();
        }
        break;
    case EnvironmentRole::PSERVER:
        //wait server done
        while (true) {
            sleep(10000);
        }
        break;
    }

    return 0;
}
