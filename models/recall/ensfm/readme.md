# ENSFM召回模型

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── sample_data
        ├── train.csv #训练数据样例
        ├── test.csv #训练数据样例
├── config.yaml # sample数据配置
├── config_bigdata.yaml # 全量数据数据配置
├── download.sh # 下载全量数据
├── dygraph_model.py # 构建动态图
├── infer.py # 预测脚本
├── movielens_reader.py #数据读取程序
├── net.py # 模型核心组网（动静统一）
├── readme.md #文档
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
ENSFM 是一个只有一层预测层的浅 FM 模型，跟 DeepFM, CFM 相比在复杂度和参数量上都更少，却在模型效果上表现显著的优势。结果验证了论文[Eicient Non-Sampling Factorization Machines for Optimal
Context-Aware Recommendation](http://www.thuir.cn/group/~mzhang/publications/TheWebConf2020-Chenchong.pdf)的观点：负采样策略并不足以使模型收敛到最优。与之相比，非采样学习对于优化 Top-N 推荐任务是非常有效的。
## 数据准备
本模型使用论文中的数据集ml-1m（即MovieLens数据集）、lastfm和frappe

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在ncf模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/recall/ensfm # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 动态图预测
python -u infer.py -m config.yaml 
```
## 模型组网
模型的总体结构如下：
<img align="center" src="picture/ensfm.jpg">

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
在全量数据下模型的指标如下：  

| 模型 | HR@5 | HR@10 | batch_size | epoch_num| Time of each epoch |
| :------| :------ |:------ | :------ | :------| :------ | 
| ENSFM | 0.058 | 0.1 | 512 | 500 | 约2分钟 |

1. 确认您当前所在目录为PaddleRec/models/recall/ensfm  
2. 进入Paddlerec/datasets/ml-1m_ensfm
3. 执行该脚本，会从国内源的服务器上下载我们预处理完成的movielens全量数据集，并解压到指定文件夹。

``` bash
cd ../../../datasets/movielens_pinterest_NCF
sh run.sh
```

```bash
cd - # 切回模型目录
# 动态图训练并得到指标(这里需要使用bash启动脚本)
python -u ../../../tools/trainer.py -m config_bigdata.yaml
python -u infer.py -m config_bigdata.yaml
```

## 进阶使用
  
## FAQ
