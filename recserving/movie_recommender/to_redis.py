#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/bin/env python

import redis
import json
import codecs

#1::Toy Story (1995)::Animation|Children's|Comedy
def process_movie(lines, redis_cli):
    for line in lines:
        if len(line.strip()) == 0:
            continue
        tmp = line.strip().split("::")
        movie_id = tmp[0]
        title = tmp[1]
        genre_group = tmp[2]
        
        tmp = genre_group.strip().split("|")
        genre = tmp
        movie_info = {"movie_id" : movie_id,
                "title" : title,
                "genre" : genre
                }
        redis_cli.set("{}##movie_info".format(movie_id), json.dumps(movie_info))

#1::F::1::10::48067
def process_user(lines, redis_cli):
    for line in lines:
        if len(line.strip()) == 0:
            continue
        tmp = line.strip().split("::")
        user_id = tmp[0]
        gender = tmp[1]
        age = tmp[2]
        job = tmp[3]
        zip_code = tmp[4]
        user_info = {"user_id": user_id,
                "gender": gender,
                "age": age,
                "job": job,
                "zip_code": zip_code
                }
        redis_cli.set("{}##user_info".format(user_id), json.dumps(user_info))

if __name__ == "__main__":
    r = redis.StrictRedis(host="127.0.0.1", port="6379") 
    with codecs.open("users.dat", "r",encoding='utf-8',errors='ignore') as f:
        lines = f.readlines()
        process_user(lines, r)
    with codecs.open("movies.dat", "r",encoding='utf-8',errors='ignore') as f:
        lines = f.readlines()
        process_movie(lines, r)

    
