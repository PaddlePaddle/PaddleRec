#pragma once

// Hide BLOG
#define BUTIL_LOGGING_H_
#define COMPACT_GOOGLE_LOG_NOTICE COMPACT_GOOGLE_LOG_INFO
#include "communicate/ps_server.h"
#include "communicate/ps_client.h"

namespace paddle {
namespace custom_trainer {
namespace feed {
    
class RuntimeEnvironment;
class PSlib {
public:
    PSlib() {}
    virtual ~PSlib() {}
    int initialize(const std::string& conf_path, 
        RuntimeEnvironment* environment);
        
    virtual paddle::ps::PSServer* ps_server();
    virtual paddle::ps::PSClient* ps_client();
    virtual paddle::PSParameter* get_param();
private:
    void init_gflag();
    virtual int init_server();
    virtual int init_client();

    paddle::PSParameter _ps_param;
    RuntimeEnvironment* _environment;
    std::shared_ptr<paddle::ps::PSServer> _server_ptr;
    std::shared_ptr<paddle::ps::PSClient> _client_ptr;  
};

}  // namespace feed
}  // namespace custom_trainer
}  // namespace paddle
