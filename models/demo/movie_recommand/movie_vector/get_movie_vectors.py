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
        self.movie_id, self.title, self.genres = "","",""
        pass

def hash2(a):
    return hash(a) % 60000000

ctr_client = LocalPredictor()
ctr_client.load_model_config("serving_server_dir")
with codecs.open("movies.dat", "r",encoding='utf-8',errors='ignore') as f:
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
    item_infos= []
    if isinstance(genre, list):
        movie.genres = genre
    else:
        movie.genres = [genre]
    movie.movie_id, movie.title = movie_id, title
    item_infos.append(movie)

    dic = {"movieid": [], "title": [], "genres": []}
    batch_size = len(item_infos)
    movie_lod = [0]
    title_lod = [0]
    genres_lod = [0]
    for i, item_info in enumerate(item_infos):
        dic["movieid"].append(hash2(item_info.movie_id))
        dic["title"].append(hash2(item_info.title))
        dic["genres"].extend([hash2(x) for x in item_info.genres])
        movie_lod.append(i+1)
        title_lod.append(i+1)
        genres_lod.append(genres_lod[-1] + len(item_info.genres))

    dic["movieid.lod"] = movie_lod
    dic["title.lod"] = title_lod
    dic["genres.lod"] = genres_lod

    for key in dic:
        dic[key] = np.array(dic[key]).astype(np.int64).reshape(len(dic[key]),1)

    fetch_map = ctr_client.predict(feed=dic, fetch=["save_infer_model/scale_0.tmp_0"], batch=True)
    ff.write("{}:{}\n".format(movie_id, str(fetch_map["save_infer_model/scale_0.tmp_0"].tolist()[0])))
ff.close()
