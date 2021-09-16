
# Paddle Inference Golang API Benchmark and Demo

1. 准备
Golang inference demo 的编译和安装依赖 capi 和库，需要安装好 c 预测库，完整教程请参考
[link](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/inference/goapi) 安装。

> 需要完成设置 paddle_inference_c 和 LD_LIBRARY_PATH

2. 编译

```shell
go mod init demo

go build benchmark.go

```

3. 运行测试

```shell
./benchmark --help # 查看命令参数
```

使用参考 run 脚本，按需求修改好之后使用如下命令启动

```shell
bash run
```




