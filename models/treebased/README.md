# Paddle TDM系列模型解决方案

以下是本例的简要目录结构及说明：
```
├── treebased
|   ├── builder                      生成树索引目录
|   |   ├── tree_index_builder.py    生成树索引脚本
|   |   ├── get_item.sh              下载建树所需item文件
|   ├── data                         数据生成目录
|   |   ├── data_cutter.py           数据切分脚本
|   |   ├── data_generator.py        样本生成脚本
|   |   ├── demo_train_data          demo训练集目录
|   |   |   ├──train_data            demo训练集
|   |   ├── demo_test_data           demo测试集目录
|   |   |   ├──test_data             demo测试集
|   ├── data_prepare.sh              下载数据及启动数据切分、样本生成脚本
|   ├── jtm                          JTM算法目录
|   ├── tdm                          TDM算法目录
|   |   ├── config.yaml              模型配置
|   |   ├── config_ub.yaml           全量数据模型配置
|   |   ├── get_leaf_embedding.py    从已训好模型中抽取Item对应的embedding
|   |   ├── infer.py                 预测脚本
|   |   ├── model.py                 模型核心组网
|   |   ├── reader.py                数据读取
|   |   ├── static_model.py          构建静态图
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

本示例代码提供了基于PaddlePaddle实现的树模型推荐搜索算法，包括[TDM](https://arxiv.org/pdf/1801.02294.pdf)，[JTM](https://arxiv.org/pdf/1902.07565.pdf)。树模型是为大规模推荐系统设计的、能承载任意先进模型来高效检索用户兴趣的推荐算法解决方案。该方案基于树结构，提出了一套对用户兴趣度量进行层次化建模与检索的方法论，使得系统能直接利高级深度学习模型在全库范围内检索用户兴趣。其基本原理是使用树结构对全库item进行索引，然后训练深度模型以支持树上的逐层检索，从而将大规模推荐中全库检索的复杂度由O(n)（n为所有item的量级）下降至O(log n)。

## 数据准备

目前提供了特定数据下的demo训练和测试数据；训练集中，前128位为浮点型数据，代表了历史物品对应的bert向量；最后一个整型数据为item id，代表用户会点击该物品。

## 运行环境

PaddlePaddle>=2.0

python 3.7

os : linux

## 快速开始

基于demo数据集，快速上手TDM系列模型，为您后续设计适合特定使用场景的模型做准备。

假定您PaddleRec所在目录为${PaddleRec_Home}。

- Step1: 进入treebased/builde文件夹下，完成item文件下载、建树等准备工作。

```shell
cd ${PaddleRec_Home}/models/treebased/builder
./get_item.sh
python tree_index_builder.py --mode by_kmeans --input item_mini.txt --output ./tree.pb
```
执行上述命令之后，您会在builder文件夹下面看到如下结构：

```
├── builder
|   ├── tree_index_builder.py    生成树索引脚本
|   ├── get_item.sh              下载建树所需item文件
|   ├── tree_emb.npy             树中各个节点的向量表示 
|   ├── ids_id.txt               用于将item id转换成模型可识别id，具体细节可见tree_index_builder.py
|   ├── tree.pb                  预处理后，生成的初始化树文件
```

- Step2: 训练。config.yaml中配置了模型训练所有的超参，运行方式同PaddleRec其他模型静态图运行方式。当前树模型暂不支持动态图运行模式。

```shell
cd tdm
python -u ../../../tools/static_trainer.py -m config.yaml 
```

- Step3: 预测，命令如下所示。其中第一个参数为训练config.yaml位置，第二个参数为预测模型地址。

```
python infer.py config.yaml ./output_model_tdm_demo/0/
```

- Step4: 提取Item（叶子节点）的Embedding，用于重新建树，开始下一轮训练。命令如下所示，其中第一个参数为训练config.yaml位置，第二个参数模型地址，第三个参数为输出文件名称。

```
python get_leaf_embedding.py config.yaml  ./output_model_tdm_demo/0/ epoch_0_item_embedding.txt
```

- Step5: 基于Step4得到的Item的Embedding，重新建树。命令如下所示。

```
cd ../builder && python tree_index_builder.py --mode by_kmeans --input ../tdm/epoch_0_item_embedding.txt --output ./new_tree.pb
```

- Step6: 修改config.yaml中tree文件的路径为最新tree.pb，返回Step2，开始新一轮的训练。

# 模型组网
模型的组网本质是一个二分类任务，代码参考`model.py`。模型将用户历史点击序列id与目标item concat起来送入dnn结构，可选attention结构。

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。由于数据量较大，如果需要复现readme中的效果，需要多节点训练。
在全量数据下模型的指标如下：  
| 模型 | 建树 | recall_rate | precision_rate | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------| :------ | :------ | :------ | 
| TDM | 首次建树 | 0.48% | 0.11% | 3000 | 1 | 约5.5小时 |
| TDM | 基于首轮模型重新建树 | 1.0% | 0.25% | 3000 | 1 | 约5.5小时 |

1. 参考[快速开始](#快速开始)Step1，下载全量数据并进行处理
```
./data_prepare.sh user_behavior
```
2. 剩余步骤参考[快速开始](#快速开始)Step2-6
```
注意替换：config.yaml->config_ub.yaml、output_model_tdm_demo->output_model_tdm_ub、demo_data->ub_data_new
```
## 进阶使用
  
## FAQ
