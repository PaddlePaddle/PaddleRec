#include <fcntl.h>
#include <fstream>
#include <sstream>
#include "json2pb/json_to_pb.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "paddle/fluid/train/custom_trainer/feed/common/pslib_warpper.h"
#include "paddle/fluid/train/custom_trainer/feed/common/runtime_environment.h"

namespace paddle {
namespace custom_trainer {
namespace feed {

int PSlib::initialize(const std::string& conf_path, 
    RuntimeEnvironment* environment) {
    _environment = environment;
    init_gflag();    
    int file_descriptor = open(conf_path.c_str(), O_RDONLY);
    if (file_descriptor == -1){
        LOG(ERROR) << "FATAL: cant open " << conf_path;
        return -1;
    }
    google::protobuf::io::FileInputStream fileInput(file_descriptor);
    if (!google::protobuf::TextFormat::Parse(&fileInput, &_ps_param)) {
        LOG(ERROR) << "FATAL: fail to parse " << conf_path;
        return -1;
    }
    close(file_descriptor); 
    init_server();
    init_client();
    return 0;
}
        
int PSlib::init_server() {
    if (_environment->is_role(EnvironmentRole::PSERVER)) {
        _server_ptr.reset(paddle::ps::PSServerFactory::create(_ps_param));
        _server_ptr->configure(_ps_param, *(_environment->ps_environment()), 
            _environment->rank_id(EnvironmentRole::PSERVER));
        _server_ptr->start(); 
    }
    _environment->barrier(EnvironmentRole::ALL);
    _environment->ps_environment()->gather_ps_servers();
    return 0;
}

int PSlib::init_client() {
    // 所有节点都启动psclient
    _client_ptr.reset(paddle::ps::PSClientFactory::create(_ps_param));
    _client_ptr->configure(_ps_param, *(_environment->ps_environment()), 
        _environment->rank_id(EnvironmentRole::ALL));

    _environment->barrier(EnvironmentRole::ALL);
    _environment->ps_environment()->gather_ps_clients();
    _client_ptr->create_client2client_connection();
    return 0;
}

paddle::ps::PSServer* PSlib::ps_server() {
    return _server_ptr.get();
}

paddle::ps::PSClient* PSlib::ps_client() {
    return _client_ptr.get();
}

paddle::PSParameter* PSlib::get_param() {
    return &_ps_param;
}

void PSlib::init_gflag() {
    int cnt = 4;
    char** params_ptr = new char*[cnt];
    std::cout << "alloc_ptr" << params_ptr << std::flush;
    char p0[] = "exe default";
    char p1[] = "-max_body_size=314217728";
    char p2[] = "-bthread_concurrency=40";
    char p3[] = "-socket_max_unwritten_bytes=2048000000";
    params_ptr[0] = p0;
    params_ptr[1] = p1;
    params_ptr[2] = p2;
    params_ptr[3] = p3;
    // ParseCommandLineFlags would change param_ptr, so copy it
    char** params_ptrp = params_ptr;
    ::google::ParseCommandLineFlags(&cnt, &params_ptrp, true);
    delete[] params_ptr;
}

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
