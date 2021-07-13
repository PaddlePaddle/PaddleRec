# 推荐服务

## 简介
本工具用于构建一个完整的推荐服务。以电影推荐系统为例，展示了使用PaddleRec搭建一个电影推荐系统的全部流程和效果。使用者可以通过我们构建的五种服务（用户模型服务，内容模型服务，召回服务，排序服务，还有应用服务）直观的体验到一个完整的推荐系统的运作方式。我们欢迎大家前往paddlerec/models/demo/movie_recommand项目，以及AI Studio中的公开项目[《十分钟！全流程！从零搭建推荐系统》](https://aistudio.baidu.com/aistudio/projectdetail/559336)，[《PaddleRec公开教程》](https://aistudio.baidu.com/aistudio/projectdetail/1268461)上体验。

## 目录结构
以下是本例的简要目录结构及说明： 
```
├── readme.md #介绍文档
├── __init__.py
├── get_data.sh  #下载数据集和模型的脚本
├── as.py #应用服务的实现程序
├── cm.py #内容模型服务的实现程序
├── rank.py #排序服务的实现程序
├── recall.py #召回服务的实现程序
├── um.py #用户模型服务的实现程序
├── start_server.sh #启动服务的脚本
├── test_client.py #启动客户端调用服务
├── to_redis.py #将数据加载入redis
├── proto
    ├── __init__.py
    ├── run_codegen.py #将proto转换为python格式
    ├── as.proto #应用服务中发出请求获得protobuf信息的数据结构
    ├── cm.proto #内容模型服务中发出请求获得protobuf信息的数据结构
    ├── item_info.proto #电影数据的数据结构表示
    ├──rank.proto #排序服务中发出请求获得protobuf信息的数据结构
    ├──recall.proto #召回服务中发出请求获得protobuf信息的数据结构
    ├──um.proto #用户模型服务中发出请求获得protobuf信息的数据结构
    ├──user_info.proto #用户数据的数据结构表示
```

## 运行环境
linux 终端
Paddle 版本：1.8+  
Python 版本：3.6  

## 设计方案

### 推荐系统流程总览
我们提供了模型的在线服务部署，本服务部署需要在linux终端模式当中执行。这里我们给出了如图所示几种服务。
<p align="center">
<img align="center" src="./demo_framework.jpg">
<p>

### 名词解释
本工具一共启动了5个在线服务，分别是用户模型服务，内容模型服务，召回服务，排序服务，还有应用服务。  
**um**: 用户模型服务(User Model)  
**cm**：内容模型服务(Content Model)  
**recall**：召回服务  
**rank**：排序服务  
**as**：应用服务（application service）  

### 业务场景
以电影推荐系统为例，讲解如何处理数据以及每个服务的实现思路
1. 电影推荐系统使用的数据：  
[MovieLens数据集](https://grouplens.org/datasets/movielens/)是一个关于电影评分的数据集，数据来自于IMDB等电影评分网站。该数据集中包含用户对电影的评分信息，用户的人口统计学特征以及电影的描述特征，非常适合用来入门推荐系统。MovieLens数据集包含多个子数据集，为了后面能够快速展示一个完整的训练流程，教程中我们选取了其中一个较小的子数据集[ml-1m](https://grouplens.org/datasets/movielens/1m/)，大小在1M左右。该数据集共包含了6000多位用户对近3900个电影的100多万条评分数据，评分为1～5的整数，其中每个电影的评分数据至少有20条，基本可以满足教学演示的需求。该数据集包含三个数据文件，分别是：
- users.dat：存储用户属性信息的txt格式文件，格式为“UserID::Gender::Age::Occupation”，其中原始数据中对用户年龄和职业做了离散化处理。      
    
| user_id | 性别 | 年龄 | 职业 |      
| -------- | -------- | -------- | -------- |       
| 2181 |  男  | 25  | 自由职业 |  
| 2182 |  女  | 25  | 学生 |          
| 2183 |  女  | 56  | 教师 |   
| ... |  ...  | ...  | ... |           

- movies.dat：存储电影属性信息的txt格式文件，格式：“MovieID::Title::Genres”。   
     
| movie_id | title | 类别 |      
| -------- | -------- | -------- |      
| 260     |Star Wars:Episode IV(1977)| 动作，科幻  |      
| 1270    | Three Amigos!(1986)| 喜剧  |    
| 2763    | Thomas Crown Affair,The(1999)| 动作，惊悚 |     
| 2858    | American Beauty(1999)| 喜剧  |     
| ...   | ... | ... |    
     
- ratings.dat：存储电影评分信息的txt格式文件，格式：“UserID::MovieID::Rating::time”。  
               
| user_id | movie_id | 评分 | 评分时间 |   
| ----- | -------- | -------- | -------- |      
| 2181   | 2858    | 4分     | 974609869 | 
| 2181   | 260    | 5分     | 974608048 |     
| 2181   | 1270    | 5分     | 974666558 |    
| 2182   | 3481    | 2分     | 974607734 |     
| 2183   | 2605   | 3分     | 974608443 |       
| 2183   | 1210    | 4分     | 974607751 |      
| ...   | ...    | ...     |   ...     |    

2. 用户模型服务和内容模型服务的实现思路  
用户模型和内容模型分别使用了数据集当中的users.dat和movies.dat数据，我们会首先使用get_data.sh脚本获取数据集，在to_redis.py中将数据经过解析之后保存在redis当中，用户模型以user_id作为key，内容模型以movie_id作为key。用户模型服务和内容模型服务的逻辑分别是从redis当中按照用户传入的key来寻找对应的value并封装成protobuf结果返回。  
3. 召回服务的实现思路  
召回服务使用离线计算把每个用户的召回结果计算好，我们会首先使用get_data.sh脚本下载计算好的结果，再将结果灌入redis中。最后在使用召回服务时，从redis当中按照传入的用户id来寻找对应的评分结果并封装成protobuf返回。但是由于数据量过大，如果将全部用户的召回数据加载入redis中，需要花费很长时间。因此我们只加载了user_id为1-13的用户的召回结果，方便大家快速体验。  
4. 排序服务的实现思路
排序服务是用PaddleRec训练好的CTR模型，用Paddle Serving启动来提供预测服务能力，用户传入一个用户信息和一组内容信息，接下来就能经过特征抽取和排序计算，求得最终的打分，按从高到低排序返回给用户。  
5. 应用服务的实现思路
应用服务就是以上流程的串联，设计的流程是用户传入自己的user id，查找到对应的用户模型，再从召回服务中得到召回movie列表，接下来查询内容模型得到列表中所有电影的信息，最终两个结合在排序服务中得到所有候选电影的从高到低的打分，最终还原成原始的电影信息返回给用户。  

### 支持的输入范围
目前支持的输入为：  
um服务：支持查询的user_id为1-6040  
cm服务：支持查询的movie_id为1-3952  
recall服务：支持召回user_id为1-13的用户排分前100名的电影  
rank服务：默认输出user_id为1的用户的排序结果，可以支持对user_id范围内任意用户和movie_id范围内任意个数的电影的打分和排序。由于rank服务参数较多，如需定制可参考代码自行改动。  
as服务：受限于recall服务，目前只支持user_id为1-13的用户跑通全流程。  

## 如何使用
### 快速启动步骤
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
目前milvus需要用docker远端启动，在宿主机上启动。

```
# 下载配置文件
mkdir -p /home/$USER/milvus/conf
cd /home/$USER/milvus/conf
wget https://raw.githubusercontent.com/milvus-io/milvus/v1.0.0/core/conf/demo/server_config.yaml

# 启动 Milvus 服务
sudo docker run -d --name milvus_cpu_1.0.0 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:1.0.0-cpu-d030521-1ea92e

# 安装 Milvus Python SDK
pip install pymilvus==1.0.1
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
python test_client.py um 5 # 查询user-id 为5的用户信息
python test_client.py cm 5 # 查询movie-id 为5的电影信息
python test_client.py recall 5 # demo召回服务预测，user id=5
python test_client.py rank # demo排序服务预测，由于rank服务参数较多，如需定制可参考代码
```

### 一键看到指定用户的推荐结果
提供两种模式，一种是用user.dat定义的用户id来做查询，另一种是自己定义新用户，顺序是性别（M/F），年龄（具体数字），工作（职业编号）
```
python test_client.py as 5 #根据用户id查看推荐结果
python test_client.py as M 25 17 # 根据用户信息查看推荐结果
```
as服务在经过查询用户、召回、查询电影、排序之后，根据分数降序，从大到小，把最适合该用户的电影信息返回回来，方便您一键直观的看到结果。结果中是由大到小排序的电影信息，每个电影信息包含了电影的id，电影名和影片类型。
```
item_infos {
  movie_id: "220"
  title: "Castle Freak (1995)"
  genre: "Horror"
}
item_infos {
  movie_id: "3576"
  title: "Hidden, The (1987)"
  genre: "Action, Horror, Sci-Fi"
}
item_infos {
  movie_id: "3409"
  title: "Final Destination (2000)"
  genre: "Drama, Thriller"
}
item_infos {
  movie_id: "1993"
  title: "Child\'s Play 3 (1992)"
  genre: "Horror"
}
```

### 查看每个步骤的结果
您也可以分步查看每一个环节后的输出  

python test_client.py um 5 # 查询user-id 为5的用户信息  
um服务用于查询用户的信息，您可以选择一名用户，通过该服务获得用户的id，性别，年龄，工作，邮政编码。其中error=200为HTTP状态码，200表示请求已成功，请求所希望的响应头或数据体将随此响应返回。出现此状态码是表示正常状态。示例结果如下：  
```
error {
  code: 200
}
user_info {
  user_id: "5"
  gender: "M"
  age: 25
  job: "20"
  zipcode: "55455"
}

```
python test_client.py cm 3878 # 查询movie-id 为3878的电影信息  
cm服务用于查询电影信息，您可以选择一个电影的id，通过该服务获得电影的id，电影名和影片类型。示例结果如下：  
```
error {
  code: 200
}
item_infos {
  movie_id: "3878"
  title: "X: The Unknown (1956)"
  genre: "Sci-Fi"
}
```
python test_client.py recall 5 # demo召回服务预测，user id=5  
recall服务用于根据您选择的用户id召回排分前100名的电影，显示每一个电影的id和预估分值。示例结果如下：  
```
score_pairs {
  nid: "3878"
  score: 4.4319376945495605
}
score_pairs {
  nid: "1971"
  score: 4.392200469970703
}
score_pairs {
  nid: "3375"
  score: 4.370407581329346
}
score_pairs {
  nid: "1973"
  score: 4.357224464416504
}
```
python test_client.py rank # demo排序服务预测，由于rank服务参数较多，如需定制可参考代码。示例结果如下：  
```
error {
  code: 200
}
score_pairs {
  nid: "1"
  score: 3.7467310428619385
}
```

## 如何二次开发

**在线部分**: 您如果需要使用demo展示自己的模型和数据，需要根据自己的数据格式自行更改相关的代码。  
1. 首先，您需要了解自己的数据。了解用户侧的数据结构，和内容侧的数据结构，并且找到与他们产生关联的数据结构和服务。  
比如在我们电影推荐项目的demo展示中，我们用户侧数据包含user_id，性别，年龄，工作，我们将这些信息使用proto格式写在user_info.proto中。我们内容侧的数据包含电影id，电影名，电影类别，我们将这些信息使用proto格式写在item_info.proto中。您需要根据自己的数据，定义自己的数据结构。  
2. 我们在电影推荐项目的demo展示中，在to_redis.py中将数据集存储入redis当中，您在使用自己的数据集时，需要自行更改其中的数据处理逻辑，将您的数据输入redis中。  
3. 我们在电影推荐项目的demo展示中，共定义了五种服务，分别是用户模型，内容模型，召回模型，排序模型，和应用服务。提供了查询用户的信息，查询电影信息，召回电影，按打分排序和一键跑通流程的功能。功能的实现分别在um.py，cm.py，recall.py，rank.py，as.py中定义。您在实现自己的服务时，可以参考这几个服务，定义自己的服务的逻辑实现。  
4. 定义各种服务请求获得数据的数据结构，如电影推荐项目的五种服务请求的数据结构放在um.proto，cm.proto，recall.proto，rank.proto，as.proto当中。您可以根据您的数据和服务实现，定义服务中每次发出请求获得怎样的数据。  
5. 最后在test_client.py中定义在客户端中，如何调用服务，传递参数。  

**离线部分**: 
1. 获得Recall所需模型
在`models/demo/movie_recommand`下分别执行
```
python3 -u ../../../tools/static_trainer.py -m recall/movie.yaml
python3 -u ../../../tools/static_trainer.py -m recall/user.yaml
```
训练好的user/movie模型首先需要参照[Paddle保存的预测模型转为Paddle Serving格式可部署的模型](https://github.com/PaddlePaddle/Serving/blob/develop/doc/SAVE_CN.md)

2. 获得用于milvus建库的电影向量文件
`movie.yaml`训练所保存的模型可以用于生成全库的电影向量。需要将数据 movie.dat 复制一份到 get_movie_vector.py 同一目录下，在运行的时候需要直接读取数据集。此外 serving_service 也需要和 get_movie_vector.py 放在同一级目录。运行
```
python3 get_movie_vectors.py
```
获得movie端embedding配送文件，该文件用于milvus建库。

注：user端的模型，直接用于`recall.py`的用户向量预测。 

3. 获得rank模型
在`models/demo/movie_recommand`下执行
```
python3 -u ../../../tools/static_trainer.py -m rank/config.yaml
```
可以得到排序模型。转换成Serving格式可部署模型后，可以用于`rank.py`的排序模型。
