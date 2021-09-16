// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../include/utils.h"

void stringTokenize(const char *str, const char *tokens, std::vector<std::string> &ret)
{
    char *tmp = new char[strlen(str) + 2];
    char *saved_ptr = NULL;
    strncpy(tmp, str, strlen(str));
    tmp[strlen(str)] = '\0';
    char *token = strtok_r(tmp, tokens, &saved_ptr);
    while (token)
    {
        ret.push_back(token);
        token = strtok_r(NULL, tokens, &saved_ptr);
    }

    delete[] tmp;
    return;
}
