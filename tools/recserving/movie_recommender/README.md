## Movie Recommender Demo Serving 

### 介绍

本实例是将PaddleRec下的movie_recommender demo上线Serving，展示推荐系统各个组件的工作原理。

### 前提条件
Paddle 版本：1.8+
Python 版本：2.7/3.6

### 名词解释

**um**: 用户模型(User Model)
**mm**：电影模型(Movie Model)
**recall**：召回
**rank**：精排序
**gr**：通用排序服务

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

2. redis服务启动

```
wget http://download.redis.io/releases/redis-stable.tar.gz --no-check-certificate
tar -xf redis-stable.tar.gz && cd redis-stable/src && make && ./redis-server &
```

3. 运行相关命令
```
sh get_data.sh
sh start_server.sh
```

4. 运行客户端
```
export PYTHONPATH=$PYTHONPATH:$PWD/proto
python test_client.py gr 5 # 获得 5号用户的推荐信息
```

这里我们给出了几种模式。
```
python test_client.py um 5 # 查询user-id 为1的用户信息
python test_client.py mm 5 # 查询movie-id 为1的电影信息
python test_client.py recall 5 # demo召回服务预测，user id=5
python test_client.py rank # demo排序服务预测，由于rank服务参数较多，如需定制可参考代码
```
