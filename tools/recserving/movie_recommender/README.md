## Movie Recommender Demo Serving 

### 介绍

本实例是将PaddleRec下的movie_recommender demo上线Serving，展示推荐系统各个组件的工作原理。

### 前提条件
Paddle 版本：1.8+
Python 版本：2.7/3.6

### 名词解释

**um**: 用户模型(User Model)
**cm**：内容模型(Content Model)
**recall**：召回
**rank**：排序
**as**：应用服务(App Service)

### 操作步骤

1. 安装相关库
```
pip install -U redis pyyaml grpcio-tools
pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server-0.0.0-py2-none-any.whl https://paddle-serving.bj.bcebos.com/whl/paddle_serving_client-0.0.0-cp27-none-any.whl https://paddle-serving.bj.bcebos.com/whl/paddle_serving_app-0.0.0-py2-none-any.whl
```
由于paddle serving最新版本还在测试，因此还需要用开发版的whl包，对于其他环境，可以参考如下链接。
```
https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md
```

2. redis/milvus服务启动

```
wget http://download.redis.io/releases/redis-stable.tar.gz --no-check-certificate
tar -xf redis-stable.tar.gz && cd redis-stable/src && make && ./redis-server &
```
目前milvus需要用docker远端启动，在宿主机上启动

```
sudo docker run -d --name milvus_cpu_0.11.0 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:0.11.0-cpu-d101620-4c44c0
```

3. 运行相关命令
```
sh get_data.sh
sh start_server.sh
```

4. 运行客户端
```
export PYTHONPATH=$PYTHONPATH:$PWD/proto
python test_client.py as 5 # 获得 5号用户的推荐信息
```

这里我们给出了几种模式。
```
python test_client.py um 5 # 查询user-id 为1的用户信息
python test_client.py cm 5 # 查询movie-id 为1的电影信息
python test_client.py recall 5 # demo召回服务预测，user id=5
python test_client.py rank # demo排序服务预测，由于rank服务参数较多，如需定制可参考代码
```

### 附录
获得Rank模型和Recall模型
在`models/demo/movie_recommand`下分别执行
```
python3 -m paddlerec.run -m recall/user.yaml
python3 -m paddlerec.run -m recall/movie.yaml
```
训练好的user/movie模型首先需要参照[Paddle保存的预测模型转为Paddle Serving格式可部署的模型](https://github.com/PaddlePaddle/Serving/blob/develop/doc/INFERENCE_TO_SERVING_CN.md)

接下来运行
```
python3 get_movie_vectors.py 
```
获得movie端embedding配送文件，该文件用于milvus建库。

user端的模型，直接用于`recall.py`的用户向量预测。

对于rank模型，在`models/demo/movie_recommand`下执行
```
python3 -m paddlerec.run -m rank/config.yaml
```
可以得到排序模型。转换成Serving格式可部署模型后，可以用于`rank.py`的排序模型。


