# 分布式模式介绍

当模型、数据的规模达到单机训练的瓶颈之后，分布式训练是必然选择。目前PaddleRec可提供两种分布式训练的模式。  
参数服务器：推荐系统领域常用的并行训练方式，ParameterServer模式提供了基于参数服务器的分布式训练功能 。  
GPU多机训练：如果您希望使用GPU进行多机多卡训练，Collective模式提供了使用飞桨进行单机多卡，多机多卡训练的功能。  
本教程讲解如何使用以上两种模式，如果您希望深入学习paddle的分布式训练功能，建议您访问[分布式深度学习介绍](ps_background.md)进行深入了解

## 版本要求
在编写分布式训练程序之前，用户需要确保已经安装paddlepaddle-2.0.0-rc-cpu或paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架。  

## ParameterServer模式
为了提高模型的训练效率，分布式训练应运而生，其中基于参数服务器的分布式训练为一种常见的中心化共享参数的同步方式。与单机训练不同的是在参数服务器分布式训练中，各个节点充当着不同的角色：  
训练节点：该节点负责完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。  
服务节点：在收到所有训练节点传来的梯度后，该节点会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。  
在PaddleRec上使用ParameterServer模式启动分布式训练只需两步：
1. 在yaml配置中添加分布式相关的参数
2. 决定需要使用的训练节点和服务节点数量，并在启动命令中输入相关配置。

### 添加yaml配置
使用ParameterServer模式相较单机模式需要添加一些相关配置，在模型文件中的`config.yaml`中添加如下增量配置：
```yaml
runner:
    # 通用配置不再赘述，想了解可查看进阶教程
    sync_mode: "async" # 可选, string: sync/async/geo
    geo_step: 400 # 可选, int, 在geo模式下控制本地的迭代次数
    split_file_list: False # 可选, bool, 若每个节点上都拥有全量数据，则需设置为True 
    thread_num: 1 # 多线程配置

    # reader类型，分布式下推荐QueueDataset
    reader_type: "QueueDataset" # DataLoader / QueueDataset / RecDataset
    pipe_command: "python benchmark_reader.py" # QueueDataset 模式下的数据pipe命令
    dataset_debug: False # QueueDataset 模式下 Profiler开关
```

### 单机模拟分布式训练启动命令
在没有多台机器之前，可以使用单机模拟分布式模式，使用ParameterServer进行训练。  
下面以dnn模型为例，展示如何启动训练，支持在任意目录下运行，以下命令默认在PaddleRec根目录中运行：  
```shell
fleetrun --worker_num=1 --server_num=1 tools/static_ps_trainer.py -m models/rank/dnn/config.yaml
```

### 分布式训练启动命令
- 首先确保各个节点之间是联通的，相互之间通过IP可访问
- 在每个节点上都需要持有代码与数据
- 在每个节点上执行如下命令, 以下命令默认在PaddleRec根目录中运行
```shell
fleetrun --workers="ip1:port1,ip2:port2...ipN:portN" --servers="ip1:port1,ip2:port2...ipN:portN" tools/static_ps_trainer.py -m models/rank/dnn/config.yaml
```

## Collective模式
如果您希望可以同时使用多张GPU，快速的训练您的模型，可以尝试使用`单机多卡`或`多机多卡`模式运行。
在PaddleRec上使用Collective模式启动分布式训练需要四步：
1. 在yaml配置中添加分布式相关的参数
2. 修改reader划分数据集
3. 决定需要使用的gpu卡数，并设置环境变量
4. 在启动命令中输入相关配置，启动训练

### 添加yaml配置
使用Collective模式相较单机模式需要添加一些相关配置，首先需要在模型的yaml配置中，加入use_fleet参数，并把值设置成True。  
同时设置use_gpu为True    
```yaml
runner:
  # 通用配置不再赘述
  ...
  # use fleet
  use_fleet: True
  use_gpu: True
```
### 修改reader
目前我们paddlerec模型默认使用的reader都是继承自paddle.io.IterableDataset，在reader的__iter__函数中拆分文件，按行处理数据。当 paddle.io.DataLoader 中 num_workers > 0 时，每个子进程都会遍历全量的数据集返回全量样本，所以数据集会重复 num_workers 次，也就是每张卡都会获得全部的数据。您在训练时可能需要调整学习率等参数以保证训练效果。  
如果需要数据集样本不会重复，可通过paddle.distributed.get_rank()函数获取当前使用的第几张卡，paddle.distributed.get_world_size()函数获取使用的总卡数。并在reader文件中自行添加逻辑划分各子进程的数据。[paddle.io.IterableDataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dataloader/dataset/IterableDataset_cn.html#iterabledataset)的相关信息以及划分数据的示例可以点击这里获取。

### 单机多卡模式下指定需要使用的卡号
在没有进行设置的情况下将使用单机上所有gpu卡。若需要指定部分gpu卡执行，可以通过设置环境变量CUDA_VISIBLE_DEVICES来实现。  
例如单机上有8张卡，只打算用前4卡张训练，可以设置export CUDA_VISIBLE_DEVICES=0,1,2,3  
再执行训练脚本即可。

### 单机多卡训练启动命令
下面以wide_deep模型为例，展示如何启动训练,支持在任意目录下运行，以下命令默认在models/rank/wide_deep目录中运行：
```bash
# 动态图执行训练
python -m paddle.distributed.launch ../../../tools/trainer.py -m config.yaml
# 静态图执行训练
python -m paddle.distributed.launch ../../../tools/static_trainer.py -m config.yaml
```

注意：在使用静态图训练时，确保模型static_model.py程序中create_optimizer函数设置了分布式优化器。
```python
def create_optimizer(self, strategy=None):
    optimizer = paddle.optimizer.Adam(learning_rate=self.learning_rate, lazy_mode=True)
    # 通过Fleet API获取分布式优化器，将参数传入飞桨的基础优化器
    if strategy != None:
        import paddle.distributed.fleet as fleet
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(self._cost)
```

### 多机多卡训练启动命令
使用多机多卡训练，您需要另外一台或多台能够互相ping通的机器。每台机器中都需要安装paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架，同时将需要运行的paddlerec模型，数据集复制到每一台机器上。
- 首先确保各个节点之间是联通的，相互之间通过IP可访问
- 在每个节点上都需要持有代码与数据
- 在每个节点上执行如下命令  
从单机多卡到多机多卡训练，在代码上不需要做任何改动，只需再额外指定ips参数即可。其内容为多机的ip列表，命令如下所示：
```bash
# 动态图
# 动态图执行训练
python -m paddle.distributed.launch --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus 0,1,2,3,4,5,6,7 ../../../tools/trainer.py -m config.yaml
# 静态图执行训练
python -m paddle.distributed.launch --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus 0,1,2,3,4,5,6,7 ../../../tools/static_trainer.py -m config.yaml
```
