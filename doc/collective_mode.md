# Collective模式运行
如果您希望可以同时使用多张GPU，更为快速的训练您的模型，可以尝试使用`单机多卡`或`多机多卡`模式运行。

## 版本要求
用户需要确保已经安装paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架

## 设置config.yaml
首先需要在模型的yaml配置中，加入use_fleet参数，并把值设置成True。  
同时设置use_gpu为True    
```yaml
runner:
  # 通用配置不再赘述
  ...
  # use fleet
  use_fleet: True
```
## 单机多卡训练

### 单机多卡模式下指定需要使用的卡号
在没有进行设置的情况下将使用单机上所有gpu卡。若需要指定部分gpu卡执行，可以通过设置环境变量CUDA_VISIBLE_DEVICES来实现。  
例如单机上有8张卡，只打算用前4卡张训练，可以设置export CUDA_VISIBLE_DEVICES=0,1,2,3  
再执行训练脚本即可。

### 执行训练
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

## 多机多卡训练
使用多机多卡训练，您需要另外一台或多台能够互相ping通的机器。每台机器中都需要安装paddlepaddle-2.0.0-rc-gpu及以上版本的飞桨开源框架，同时将需要运行的paddlerec模型，数据集复制到每一台机器上。
- 首先确保各个节点之间是联通的，相互之间通过IP可访问
- 在每个节点上都需要持有代码与数据
- 在每个节点上执行命令  
从单机多卡到多机多卡训练，在代码上不需要做任何改动，只需再额外指定ips参数即可。其内容为多机的ip列表，命令如下所示：
```bash
# 动态图
# 动态图执行训练
python -m paddle.distributed.launch --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus 0,1,2,3,4,5,6,7 ../../../tools/trainer.py -m config.yaml
# 静态图执行训练
python -m paddle.distributed.launch --ips="xx.xx.xx.xx,yy.yy.yy.yy" --gpus 0,1,2,3,4,5,6,7 ../../../tools/static_trainer.py -m config.yaml
```

## 修改reader
目前我们paddlerec模型默认使用的reader都是继承自paddle.io.IterableDataset，在reader的__iter__函数中拆分文件，按行处理数据。当 paddle.io.DataLoader 中 num_workers > 0 时，每个子进程都会遍历全量的数据集返回全量样本，所以数据集会重复 num_workers 次，也就是每张卡都会获得全部的数据。您在训练时可能需要调整学习率等参数以保证训练效果。  
如果需要数据集样本不会重复，可通过 [paddle.io.get_worker_info](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dataloader/dataloader_iter/get_worker_info_cn.html#get-worker-info) 获取各子进程的信息。并在 __iter__ 函数中划分各子进程的数据。[paddle.io.IterableDataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/dataloader/dataset/IterableDataset_cn.html#iterabledataset)的相关信息以及划分数据的示例可以点击这里获取。
