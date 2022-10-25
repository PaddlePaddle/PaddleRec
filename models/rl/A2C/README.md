# 运行环境

* [PaddlePaddle with rpc](https://github.com/PaddlePaddle/Paddle/pull/45998)
* Python>3.6
* OS: Linux

# 快速开始

如下所示，在两个服务器节点上分布式运行A2C算法，其中每个节点启动四个进程，`--master`需要指定具体的ip地址。
```shell
# on node 0
python -m paddle.distributed.launch  --master ip:port --rank 0 --nnodes 2 --nproc_per_node 4 --run_mode rpc train.py

# on node 1
python -m paddle.distributed.launch  --master ip:port --rank 1 --nnodes 2 --nproc_per_node 4 --run_mode rpc train.py
```
