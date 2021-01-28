# 分布式模式介绍

当模型、数据的规模达到单机训练的瓶颈之后，分布式训练是必然选择。目前PaddleRec可直接支持基于参数服务器模式(ParameterServer)的分布式训练。


## yaml 配置

分布式模式的相关配置在模型文件中的`config.yaml`中配置，详细的yaml配置说明参考进阶教程。分布式模式相较单机模式，增量的配置如下：

```yaml
runner:
    sync_mode: "async" # 可选, string: sync/async/geo
    geo_step: 400 # 可选, int, 在geo模式下控制本地的迭代次数
    split_file_list: False # 可选, bool, 若每个节点上都拥有全量数据，则需设置为True 
    thread_num: 1 # 多线程配置

    # reader类型，分布式下推荐QueueDataset
    reader_type: "QueueDataset" # DataLoader / QueueDataset / RecDataset
    pipe_command: "python benchmark_reader.py" # QueueDataset 模式下的数据pipe命令
    dataset_debug: False # QueueDataset 模式下 Profiler开关
```

## 单机模拟分布式启动命令

支持在任意目录下运行，以下命令默认在PaddleRec根目录中运行

```shell
fleetrun --worker_num=1 --server_num=1 tools/static_ps_trainer.py -m models/rank/dnn/config.yaml
```

## 分布式环境启动命令

- 首先确保各个节点之间是联通的，相互之间通过IP可访问
- 在每个节点上都需要持有代码与数据
- 在每个节点上执行如下命令, 以下命令默认在PaddleRec根目录中运行

```shell
fleetrun --workers="ip1:port1,ip2:port2...ipN:portN" --servers="ip1:port1,ip2:port2...ipN:portN" tools/static_ps_trainer.py -m models/rank/dnn/config.yaml
```