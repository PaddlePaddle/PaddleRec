# AITM模型的点击率预估模型

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── sample_data #样例数据
        ├── train 
            ├── train.csv #训练数据样例
        ├── test 
            ├── test.csv #训练数据样例
├── __init__.py
├── config.yaml # sample数据配置
├── config_bigdata.yaml # 全量数据配置
├── reader.py #数据读取程序
├── dygraph_model.py # 构建动态图
├── net.py # 模型核心组网
├── trainer.py # 训练脚本
├── infer.py # 训练脚本
├── readme.md #文档
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介
在推荐场景里，用户的转化链路往往有多个中间步骤（曝光->点击->转化），而有些行业转化链路很长，如金融-信用卡业务，它包括曝光->点击->表单（application）->信用核准（approval）->信用卡激活（activation）。处于链路后端的节点（如approval/activation），因为转化时间久，获取难度较大，导致转化数据少，训练时类别不平衡的问题很严重。

作者设计了一种多任务模型框架，充分利用了链路上各个节点的样本，提升模型对后端节点转化率的预估
## 数据准备

数据为[Ali-CCP click](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408)
在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分。

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在aitm模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/multitask/aitm # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml

# 动态图预测
python -u ../../../tools/infer.py -m config.yaml
``` 
## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。
在全量数据下模型的指标如下：  
| 模型 | click auc | purchase auc |batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------ | :------| :------ | 
| aitm | 0.6130 |0.6166 | 2000 | 6| 约3小时 |

1. 确认您当前所在目录为PaddleRec/models/multitask/aitm
2. 进入Paddlerec/datasets/ali-cpp_aitm
3. 执行命令运行全量数据

``` bash
cd ../../../datasets/ali-cpp_aitm
sh run.sh
```
```bash
cd - # 切回模型目录
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml 
python -u ../../../tools/infer.py -m config_bigdata.yaml
```
## 进阶使用
  
## FAQ
