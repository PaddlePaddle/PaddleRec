<h1 align="center">PaddleRec inference C++（线上）</h1>
本目录是 paddlerec C++ 线上预测完整 demo 工程，包括 debug、基于 cube 的 kv 查询等功能，适用 os 为 linux：

<h2>paddlerec 中引入 cube 的背景</h2>

成熟的推荐系统的排序算法通常都有规模相当大的 embedding 表，这些 embedding 表在单台机器上存放不下，因此通常的做法是 embedding 拆解存放在外部 KV 服务上。预测时，大规模稀疏参数不需要存放在本地，而是直接去远端查询，避免了本地 lookup_table 操作。

<h2>代码目录</h2>

```
tools/inference/cpp         
|-- main.cpp # 工程 main 文件    
|-- src   # cpp 文件  
|-- include   # h 文件  
|-- proto   # proto 文件及对应的 pb 文件  
|-- paddle_inference   # paddle inference 库及依赖的第三方库  
|-- build   # 工程编译目录  
|-- bin   # 工程输出可执行文件目录  
|-- data  # 模型、参数文件及输入样例数据  
|-- cube_app  #  cube 功能相关工具、配置文件及样例 sequence file 文件  
|-- keys   # kv 功能样例 key 文件  
|-- user.flags   # 命令行参数配置文件  
|-- run.sh  # 启动脚本
|-- 其他

```

<h2>准备工作</h2>

1. 下载 PaddlePaddle C++ 预测库：[C++ 预测库下载地址](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html)，下载后解压到 tools/inference/cpp 目录下
2. 引入百度 brpc 库: [brpc 源码地址](https://github.com/apache/incubator-brpc)，下载后编译，编译输出文件放入 ./paddle_inference 目录下
3. 训练保存好的模型和参数文件、推理样本文件

<h2>cube kv 功能流程</h2>

1. 准备 Sequence File [样例seqfile](https://paddle-serving.bj.bcebos.com/others/part-000-00000 )  
2. 编译 Paddle Serving server，获取 cube、cube-builder 可执行文件(本代码仓中已提供)
3. 生成分片文件
```
    ./cube_app/cube-builder -dict_name=test_dict -job_mode=base -last_version=0 -cur_version=0 -depend_version=0 -input_path=./cube_app/cube_model -output_path=${PWD}/cube_app/data -shard_num=1 -only_build=false  
```
此处可能需要用户升级 libcurl(apt-get install libcurl3)
4. 本地配送
```
    mv ./cube_app/data/0_0/test_dict_part0/* ./cube_app/data/
    cd cube_app && ./cube 
```
5. 用户查询测试功能，inference代码中用不到
```
    ./cube-cli -dict_name=test_dict -keys [keys文件] -conf ./cube_app/cube.conf # 本项目中 cube client 的功能已经集成在 inference 代码里了
```  

<h2>测试用例</h2>

1. 不启用 cube 服务：全量带 embedding 的模型
2. 启用 cube 服务：裁剪后不带 embedding 的模型（small_model, 300 个 slot）、输入数据（demo_10_300）、可打印的varname（all_vars_small_model.txt）

<h2>编译运行</h2>

我们准备了 run.sh 脚本提供一键编译并运行，主要启动参数如下：
|        名称         |    类型    |             取值             | 是否必须 |                               作用描述                               |
| :-----------------: | :-------: | :--------------------------: | :-----: | :------------------------------------------------------------------: |
|       --modelFile        |    string    |       任意路径         |    是    |                            模型文件                           |
|       --paramFile        |    string    |       任意路径         |    是    |                            参数文件                           |
|       --trainingFile        |    string    |       任意路径         |    是    |                            样例输入数据                           |
|       --debug        |    bool    |       true/false         |    否    |                            是否使用debug功能                            |
|       --threadNum        |    int32    |       >=1         |    是    |                            预测线程个数                            |
|       --batchSize        |    int32    |       >= 1         |    是    |                            批预测样本数量                            |
|       --withCube        |    bool    |       true/false         |    否    |                            测试是否使用 cube 功能                            |
|       --cube_batch_size        |    uint64    |       >=1         |    否    |                            cube 批量查询大小                            |
|       --cube_thread_num        |    int32    |       >=1         |    否    |                            cube 查询线程数                            |
|       --config_file        |    string    |       任意路径         |    否    |                            cube client 参数配置                            |
|       CUBE        |    编译宏    |       开启或者关闭         |    否    |                            和 --withCube 一起使用，需要修改对应的模型和参数文件                            |


<h2>Notice</h2>
 
* 样例中的模型文件由 PaddlePaddle develop 分支生成的  
* paddle inference 预测库默认版本 2.1   
* cube client 代码基于 Paddle Serving v0.3.0 server  
* cube-builder、cube、cube-transfer、cube-agent、brpc 由 Paddle Serving v0.6.2 server 编译出来的，当然 v0.3.0 也可以编出来
* 由于该项目能加载完整模型和裁剪后的模型，测试时注意修改模型文件路径（在 user.flags 文件中）和 CMakeLists.txt 中的 -DCUBE 开关
