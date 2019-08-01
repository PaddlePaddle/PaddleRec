#include "paddle/fluid/train/custom_trainer/feed/executor/executor.h"

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/platform/enforce.h"


namespace paddle {
namespace custom_trainer {
namespace feed {

namespace {

int ReadBinaryFile(const std::string& filename, std::string* contents) {
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    if (!fin) {
        VLOG(2) << "Cannot open file " << filename;
        return -1;
    }
    fin.seekg(0, std::ios::end);
    contents->clear();
    contents->resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(contents->at(0)), contents->size());
    fin.close();
    return 0;
}

std::unique_ptr<paddle::framework::ProgramDesc> Load(
        paddle::framework::Executor* /*executor*/, const std::string& model_filename) {
    VLOG(3) << "loading model from " << model_filename;
    std::string program_desc_str;
    if (ReadBinaryFile(model_filename, &program_desc_str) != 0) {
        return nullptr;
    }
    std::unique_ptr<paddle::framework::ProgramDesc> main_program(
            new paddle::framework::ProgramDesc(program_desc_str));
    return main_program;
}

}

struct SimpleExecutor::Context {
    Context(const ::paddle::platform::Place& place) : place(place), executor(place) {
    }
    const ::paddle::platform::Place& place;
    ::paddle::framework::Executor executor;
    ::std::unique_ptr<::paddle::framework::ProgramDesc> main_program;
    ::std::unique_ptr<framework::ExecutorPrepareContext> prepare_context;
    details::TensorArrayBatchCleaner tensor_array_batch_cleaner;
};


SimpleExecutor::SimpleExecutor() {

}

SimpleExecutor::~SimpleExecutor() {

}

int SimpleExecutor::initialize(YAML::Node exe_config,
        std::shared_ptr<TrainerContext> context_ptr) {
    
    paddle::framework::InitDevices(false);
    if (exe_config["num_threads"]) {
        paddle::platform::SetNumThreads(exe_config["num_threads"].as<int>());
    } else {
        paddle::platform::SetNumThreads(1);
    }

    if (!exe_config["startup_program"] || 
        !exe_config["main_program"]) {
        VLOG(2) << "fail to load config";
        return -1;
    }

    try {
        _context.reset(new SimpleExecutor::Context(context_ptr->cpu_place));
        auto startup_program = Load(&_context->executor, exe_config["startup_program"].as<std::string>());
        if (startup_program == nullptr) {
            VLOG(2) << "fail to load startup_program: " << exe_config["startup_program"].as<std::string>();
            return -1;
        }
        
        _context->executor.Run(*startup_program, this->scope(), 0, false, true);

        _context->main_program = Load(&_context->executor, exe_config["main_program"].as<std::string>());
        if (_context->main_program == nullptr) {
            VLOG(2) << "fail to load main_program: " << exe_config["main_program"].as<std::string>();
            return -1;
        }
        _context->prepare_context = _context->executor.Prepare(*_context->main_program, 0);
        _context->executor.CreateVariables(*_context->main_program, this->scope(), 0);
    } catch (::paddle::platform::EnforceNotMet& err) {
        VLOG(2) << err.what();
        _context.reset(nullptr);
        return -1;
    }

    return 0;
}

int SimpleExecutor::run() {
    if (_context == nullptr) {
        VLOG(2) << "need initialize before run";
        return -1;
    }
    try {
        _context->executor.RunPreparedContext(_context->prepare_context.get(), this->scope(),
                                false, /* don't create local scope each time*/
                                false /* don't create variable each time */);

        // For some other vector like containers not cleaned after each batch.
        _context->tensor_array_batch_cleaner.CollectNoTensorVars(this->scope());
        _context->tensor_array_batch_cleaner.ResetNoTensorVars();
    } catch (::paddle::platform::EnforceNotMet& err) {
        VLOG(2) << err.what();
        return -1;
    }
    return 0;
}
    
}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
