<h1 align="center">PaddleRec inference C++（线上）</h1>
本目录是 paddlerec C++ 线上预测完整 demo 工程，包括 debug、基于 cube 的 kv 查询：
<h2>代码目录</h2>

```
inference_c++2.0         
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
```
<h2>启动脚本</h2>

```
sh run.sh
```
<h2>kv 功能</h2>

1. 准备 Sequence File [样例seqfile](https://paddle-serving.bj.bcebos.com/others/part-000-00000 )  
2. 编译 Paddle Serving server，获取 cube、cube-builder 可执行文件
3. 生成分片文件
```
    ./cube_app/cube-builder -dict_name=test_dict -job_mode=base -last_version=0 -cur_version=0 -depend_version=0 -input_path=./cube_app/cube_model -output_path=${PWD}/cube_app/data -shard_num=1 -only_build=false  
```
4. 本地配送
```
    mv ./cube_app/data/0_0/test_dict_part0/* ./cube_app/data/
    cd cube_app && ./cube 
```
5. 查询
```
    ./cube-cli -dict_name=test_dict -keys [keys文件] -conf ./cube_app/cube.conf # 本项目中 cube client 的功能已经集成在 inference 代码里了
```  

<h2>Notice</h2>
 
* 样例中的模型文件由 Paddle 2.0.2 生成的  
* paddle inference 预测库默认版本 2.1   
* cube client 代码基于 Paddle Serving v0.3.0 server  
* cube-builder、cube、cube-transfer、cube-agent、brpc 由 Paddle Serving v0.6.2 server 编译出来的，当然 v0.3.0 也可以编出来
* 由于该项目能加载完整模型和裁剪后的模型，测试时注意修改模型文件路径（在 user.flag 文件中）和 infer.h 中的 dataType（分别是 int64_t 和 float）
