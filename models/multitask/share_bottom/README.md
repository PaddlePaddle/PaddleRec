# ShareBottom

**[AI Studio在线运行环境](https://aistudio.baidu.com/aistudio/projectdetail/3238943)**

以下是本例的简要目录结构及说明： 

```
├── data # 文档
		├── train #训练数据
			├── train_data.txt
		├── test  #测试数据
			├── test_data.txt
├── __init__.py 
├── README.md #文档
├── config.yaml # sample数据配置
├── config_bigdata.yaml # 全量数据配置
├── census_reader.py # 数据读取程序
├── net.py # 模型核心组网（动静统一）
├── static_model.py # 构建静态图
├── dygraph_model.py # 构建动态图
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [模型组网](#模型组网)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介
share_bottom是多任务学习的基本框架，其特点是对于不同的任务，底层的参数和网络结构是共享的，这种结构的优点是极大地减少网络的参数数量的情况下也能很好地对多任务进行学习，但缺点也很明显，由于底层的参数和网络结构是完全共享的，因此对于相关性不高的两个任务会导致优化冲突，从而影响模型最终的结果。后续很多Neural-based的多任务模型都是基于share_bottom发展而来的，如MMOE等模型可以改进share_bottom在多任务之间相关性低导致模型效果差的缺点。

我们在Paddlepaddle实现share_bottom网络结构，并在开源数据集Census-income Data上验证模型效果。

## 数据准备
我们在开源数据集Census-income Data上验证模型效果,在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分.
数据的格式如下：
生成的格式以逗号为分割点
```
0,0,73,0,0,0,0,1700.09,0,0
```

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在share_bottom模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/multitask/share_bottom # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 

# 静态图训练
python -u ../../../tools/static_trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 静态图预测
python -u ../../../tools/static_infer.py -m config.yaml 
``` 

## 模型组网
模型结构如下：
<p align="center">
<img align="center" src="../../../doc/imgs/shared-bottom.png">
<p>


## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。 
在全量数据下模型的指标如下：
| 模型 | auc_marital | batch_size | epoch_num | Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| Share_bottom | 0.99 | 32 | 100 | 约1分钟 |

1. 确认您当前所在目录为PaddleRec/models/multitask/share_bottom  
2. 进入paddlerec/datasets/census目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的census全量数据集，并解压到指定文件夹。
``` bash
cd ../../../datasets/census
sh run.sh
``` 
3. 切回模型目录,执行命令运行全量数据
```bash
cd - # 切回模型目录
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
python -u ../../../tools/infer.py -m config_bigdata.yaml # 全量数据运行config_bigdata.yaml 
```

## 进阶使用
  
## FAQ
