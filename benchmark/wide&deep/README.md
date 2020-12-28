
# Benchmark 运行指南

## 数据处理


## 单机训练
在数据处理完成后，根据需要可修改`config.py`中的各项配置（文件中的默认值为后文Benchmark的数据所使用配置）

执行以下命令进行单机训练:

```shell
python -u train.py 
```

## 分布式训练
在数据处理完成后，根据需要可修改`config.py`中的各项配置（文件中的默认值为后文Benchmark的数据所使用配置）

执行以下命令进行本地模拟分布式训练:

```shell
fleetrun --worker_num=1 --server_num=1 train.py
```

在真实集群中，在各台机器上，分别执行以下命令进行分布式训练：

```shell
fleetrun --worker_ips="ip1:port1,ip2:port2" --server_ips="ip3:port3,ip4:port4" train.py
```


## Benchmark数据

在以下机器配置： 
> 占位

docker镜像：
> 占位

我们测得单机性能：
> 占位

集群配置为：
> 占位

测得分布式性能：
> 占位
