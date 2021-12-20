# AutoFIS模型的点击率预估模型

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── sample_data #样例数据
        ├── train
            ├── train_x.npy #训练数据样例
            ├── train_y.npy #训练数据样例
├── __init__.py
├── README.md #文档
├── config.yaml # sample数据配置
├── config_bigdata.yaml # 全量数据配置
├── net.py # 模型核心组网（动静统一）
├── criteo_reader.py #数据读取程序
├── dygraph_model.py # 构建动态图
├── trainer.py # 训练脚本
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
Automatic Feature Interaction Selection in Factorization Models（点击率预测问题下因子分解机模型的自动特征交互选择模）是华为在2020kdd上提出了新的CTR预估方法。论文指出，很多CTR预估算法都需要进行特征组合，但是传统的特征组合方式都是简单的暴力组合或者人工选择，人工选择的方式依赖于先验知识，而简单的暴力组合其实对模型的性能的提升并不是总有益的，有些组合方式其实对模型的性能提升并没有多少的帮助，甚至会损害模型的性能，而且大量无效的特征组合会形成很多的参数，降低内存的利用率。根据AutoML技术，提出AutoFIS，顾名思义，就是自动去找最佳的特征组合。

## 数据准备

数据为[Criteo](http://labs.criteo.com/downloads/download-terabyte-click-log)，选择了第6-12天的数据作为训练集，低13天的数据测试集。正负样本采用后的比例约为1:1
在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分。

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在deepfm模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/rank/deepfm # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 
``` 

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。
在全量数据下模型的指标如下：  
| 模型 | auc | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| AutodeepFM | 0.78 | 2000 | 1 | 约2小时 |

1. 确认您当前所在目录为PaddleRec/models/rank/autofis
2. 参考https://github.com/renmada/rec_datasets获取数据,放到对应的文件夹。(train: data/whole_data/train, test: data/whole_data/test)
3. 切回模型目录,执行命令运行全量数据
```bash
cd - # 切回模型目录
# 动态图训练
python trainer.py -m config_bigdata.yaml # stage0：自动搜索最佳特征组合
python trainer.py -m config_bigdata.yaml -o stage=1 # stage1：训练最终名
python -u ../../../tools/infer.py -m config_bigdata.yaml -o stage=1 # 全量数据运行config_bigdata.yaml 
```
### 日志
[stage0](./logs/stage0.log)|[state1](./logs/stage1.log)|[infer](./logs/infer.log)
## 进阶使用
  
## FAQ
