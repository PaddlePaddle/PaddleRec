# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
feed_var {
  name: "movieid"
  alias_name: "movieid"
  is_lod_tensor: true
  feed_type: 0
  shape: -1
}
feed_var {
  name: "title"
  alias_name: "title"
  is_lod_tensor: true
  feed_type: 0
  shape: -1
}
feed_var {
  name: "genres"
  alias_name: "genres"
  is_lod_tensor: true
  feed_type: 0
  shape: -1
}
fetch_var {
  name: "save_infer_model/scale_0.tmp_0"
  alias_name: "save_infer_model/scale_0.tmp_0"
  is_lod_tensor: false
  fetch_type: 1
  shape: 32
}
"""

from paddle_serving_app.local_predict import LocalPredictor
import redis
import numpy as np
import codecs


class Movie(object):
    def __init__(self):
        self.movie_id, self.title, self.genres = "", "", ""
        pass


def hash2(a):
    return hash(a) % 600000


ctr_client = LocalPredictor()
ctr_client.load_model_config("serving_server")
with codecs.open("movies.dat", "r", encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

ff = open("movie_vectors.txt", 'w')

for line in lines:
    if len(line.strip()) == 0:
        continue
    tmp = line.strip().split("::")
    movie_id = tmp[0]
    title = tmp[1]
    genre_group = tmp[2]

    tmp = genre_group.strip().split("|")
    genre = tmp
    movie = Movie()
    item_infos = []
    if isinstance(genre, list):
        movie.genres = genre
    else:
        movie.genres = [genre]
    movie.movie_id, movie.title = movie_id, title
    item_infos.append(movie)

    dic = {"movieid": [], "title": [], "genres": []}
    batch_size = len(item_infos)
    for i, item_info in enumerate(item_infos):
        dic["movieid"].append(hash2(item_info.movie_id))
        dic["title"].append(hash2(item_info.title))
        dic["genres"].extend([hash2(x) for x in item_info.genres])

    if len(dic["title"]) <= 4:
        for i in range(4 - len(dic["title"])):
            dic["title"].append("0")
    dic["title"] = dic["title"][:4]
    if len(dic["genres"]) <= 3:
        for i in range(3 - len(dic["genres"])):
            dic["genres"].append("0")
    dic["genres"] = dic["genres"][:3]

    dic["movieid"] = np.array(dic["movieid"]).astype(np.int64).reshape(-1, 1)
    dic["title"] = np.array(dic["title"]).astype(np.int64).reshape(-1, 4)
    dic["genres"] = np.array(dic["genres"]).astype(np.int64).reshape(-1, 3)

    fetch_map = ctr_client.predict(
        feed=dic, fetch=["save_infer_model/scale_0.tmp_0"], batch=True)
    ff.write("{}:{}\n".format(movie_id,
                              str(fetch_map["save_infer_model/scale_0.tmp_0"]
                                  .tolist()[0])))
ff.close()
