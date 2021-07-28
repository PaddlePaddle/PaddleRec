# 分布式模式介绍

当模型、数据的规模达到单机训练的瓶颈之后，分布式训练是必然选择。目前PaddleRec可提供三种分布式训练的模式。  
参数服务器：推荐系统领域常用的并行训练方式，ParameterServer模式提供了基于参数服务器的分布式训练功能。
GPU多机训练：如果您希望使用GPU进行多机多卡训练，Collective模式提供了使用飞桨进行单机多卡，多机多卡训练的功能。 
GPU参数服务器（GPUBox）：如果您的推荐任务中稀疏参数较大，使用GPU Collective模式在性能和显存上无法满足要求时，推荐使用最新的GPU参数服务器训练方式，通过使用GPU以及CPU多级存储实现基于参数服务器的分布式训练。
本教程讲解如何使用以上三种模式，如果您希望深入学习paddle的分布式训练功能，建议您访问[分布式深度学习介绍](ps_background.md)进行深入了解

## 版本要求
在编写分布式训练程序之前，用户需要确保已经安装paddlepaddle-2.0.0-rc-cpu或paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架。

## ParameterServer模式
为了提高模型的训练效率，分布式训练应运而生，其中基于参数服务器的分布式训练为一种常见的中心化共享参数的同步方式。与单机训练不同的是在参数服务器分布式训练中，各个节点充当着不同的角色：  
训练节点：该节点负责完成数据读取、前向计算、反向梯度计算等过程，并将计算出的梯度上传至服务节点。  
服务节点：在收到所有训练节点传来的梯度后，该节点会将梯度聚合并更新参数。最后将参数发送给训练节点，开始新一轮的训练。  

根据参数更新的方式不同，可以分为同步/异步/Geo异步三种：  
同步训练：所有Worker的进度保持一致，即每训练完一个Batch后，所有Worker会上传梯度给Server，然后开始等待Server返回更新后的参数。Server在拿到所有Worker上传的梯度后，才会开始计算更新后的参数。因此在任何一个时间点，所有Worker都处于相同的训练阶段。同步训练的优势在于Loss可以比较稳定的下降，缺点是整个训练速度较慢，这是典型的木桶原理，速度的快慢取决于最慢的那个Worker的训练计算时间，因此在训练较为复杂的模型时，即模型训练过程中神经网络训练耗时远大于节点间通信耗时的场景下，推荐使用同步训练模式。  
异步训练：与同步训练不同，在异步训练中任何两个Worker之间的参数更新都互不影响。每一个Worker完成训练、上传梯度后，Server都会立即更新参数并将结果返回至相应的训练节点。拿到最新的参数后，该训练节点会立即开始新一轮的训练。异步训练去除了训练过程中的等待机制，训练速度得到了极大的提升，但是缺点也很明显，那就是Loss下降不稳定，容易发生抖动。建议在个性化推荐（召回、排序）、语义匹配等数据量大的场景使用。  
GEO异步训练：GEO异步训练是飞桨独有的一种异步训练模式，训练过程中任何两个训练节点之间的参数更新同样都互不影响，但是每个训练节点本地都会拥有完整的训练流程，即前向计算、反向计算和参数优化，而且每训练到一定的批次(Batch) 训练节点都会将本地的参数计算一次差值(Step间隔带来的参数差值)，将差值发送给服务端累计更新，并拿到最新的参数后，该训练节点会立即开始新一轮的训练。所以显而易见，在GEO异步训练模式下，Worker不用再等待Server发来新的参数即可执行训练，在训练效果和训练速度上有了极大的提升。但是此模式比较适合可以在单机内能完整保存的模型，在搜索、NLP等类型的业务上应用广泛，比较推荐在词向量、语义匹配等场景中使用。  

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

## GPU参数服务器(GPUBox)模式
如果您的推荐任务中稀疏参数较大，使用GPU Collective模式在性能和显存上无法满足要求时，推荐使用最新的GPU参数服务器训练方式。原理和使用可参考：[GPUBOX原理与使用](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/performance/heterps.html) 

在PaddleRec上使用GPUBox模式启动分布式训练需要三步：
1. 在yaml配置中添加分布式相关的参数
2. 修改reader类型
3. 修改网络使用的embedding
3. 在启动命令中输入相关配置，启动训练

### 添加yaml配置
使用GPUBox模式相较单机模式需要添加一些相关配置，首先需要在模型的yaml配置中，加入use_fleet参数，并把值设置成True。  
同时设置use_gpu为True，sync_mode模式设置为gpubox
```yaml
runner:
  # 通用配置不再赘述
  ...
  # use fleet
  use_fleet: True
  use_gpu: True
  sync_mode: "gpubox"
```
### 修改reader
目前GPUBox模式下只支持InmemoryDataset模式，您可以在yaml配置中修改reader_type
```yaml
runner:
  # 通用配置不再赘述
  ...
  reader_type: "InmemoryDataset"
  
```

### 修改网络使用的embedding
目前GPUBox模式使用的embedding接口与其他模式暂不兼容，因此可以在models/底下的net.py里修改embedding接口：
```python
def forward(self, sparse_inputs, dense_inputs):

  sparse_embs = []
  for s_input in sparse_inputs:
      if self.sync_mode == "gpubox":
          emb = paddle.fluid.contrib.sparse_embedding(
              input=s_input,
              size=[
                  self.sparse_feature_number, self.sparse_feature_dim
              ],
              param_attr=paddle.ParamAttr(name="embedding"))
      else:
          emb = self.embedding(s_input)
      emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
      sparse_embs.append(emb)

  # 其余部分省略 ....
```

### GPU单机启动命令
下面以dnn模型为例，展示如何启动训练,支持在任意目录下运行，以下命令默认在根目录下运行：
```bash
sh tools/run_gpubox.sh

```

其中run_gpubox.sh中需要关注并设置的参数有：
```bash

# set free port if 29011 is occupied
export PADDLE_PSERVERS_IP_PORT_LIST="127.0.0.1:29011"
export PADDLE_PSERVER_PORT_ARRAY=(29011)

# set gpu numbers according to your device
export FLAGS_selected_gpus="0,1,2,3,4,5,6,7"

# set your model yaml
SC="tools/static_gpubox_trainer.py -m models/rank/dnn/config_gpubox.yaml"
```
