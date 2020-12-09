"""
feed_var {
  name: "userid"
  alias_name: "userid"
  is_lod_tensor: true
  feed_type: 0
  shape: -1
}
feed_var {
  name: "gender"
  alias_name: "gender"
  is_lod_tensor: true
  feed_type: 0
  shape: -1
}
feed_var {
  name: "age"
  alias_name: "age"
  is_lod_tensor: true
  feed_type: 0
  shape: -1
}
feed_var {
  name: "occupation"
  alias_name: "occupation"
  is_lod_tensor: true
  feed_type: 0
  shape: -1
}
fetch_var {
  name: "save_infer_model/scale_0.tmp_3"
  alias_name: "save_infer_model/scale_0.tmp_3"
  is_lod_tensor: false
  fetch_type: 1
  shape: 32
}
"""
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
from paddle_serving_app.local_predict import LocalPredictor
import redis
import numpy as np
import codecs
class User(object):
    def __init__(self):
        self.user_id, self.gender, self.age, self.occupation = "","","",""
        pass

def hash2(a):
    return hash(a) % 60000000

ctr_client = LocalPredictor()
ctr_client.load_model_config("serving_server_dir")
with codecs.open("users.dat", "r",encoding='utf-8',errors='ignore') as f:
    lines = f.readlines()

ff = open("user_vectors.txt", 'w')

for line in lines:
    if len(line.strip()) == 0:
        continue
    tmp = line.strip().split("::")
    user_id = tmp[0]
    gender = tmp[1]
    age = tmp[2]
    job = tmp[3]

    user = User()
    item_infos= []
    user.user_id = user_id
    user.gender = gender
    user.age = age
    user.occupation = job
    item_infos.append(user)

    dic = {"userid": [], "gender": [], "age": [], "occupation": []}
    batch_size = len(item_infos)
    lod = [0]
    for i, item_info in enumerate(item_infos):
        dic["userid"].append(hash2(item_info.user_id))
        dic["gender"].append(hash2(item_info.gender))
        dic["age"].append(hash2(item_info.age))
        dic["occupation"].append(item_info.occupation)
        lod.append(i+1)

    dic["userid.lod"] = lod
    dic["gender.lod"] = lod
    dic["age.lod"] = lod
    dic["occupation.lod"] = lod

    for key in dic:
        dic[key] = np.array(dic[key]).astype(np.int64).reshape(len(dic[key]),1)

    fetch_map = ctr_client.predict(feed=dic, fetch=["save_infer_model/scale_0.tmp_3"], batch=True)
    ff.write("{}\n".format(str(fetch_map["save_infer_model/scale_0.tmp_3"].tolist()[0])))
ff.close()
