/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

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
