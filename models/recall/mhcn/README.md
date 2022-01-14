# Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation

**[AI Studio在线运行环境](https://aistudio.baidu.com/studio/project/partial/verify/3406375/b2db3498abdd41a39b0a994a8e95ffcb)**

以下是本例的简要目录结构及说明：

```
├── data # 样例数据
    ├── train
        ├── train.txt
    ├── test
        ├── test.txt
    ├── ratings.txt
    ├── trusts.txt
├── __init__.py
├── config.yaml           # sample 数据配置
├── config_bigdata.yaml   # 全量数据配置
├── dygraph_model.py      # 构建动态图
├── infer.py              # 评估函数
├── lastfm_reader.py      # 数据读取函数
├── net.py                # 模型核心组网（动静统一）
├── README.md             # 文档
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

[Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation](https://arxiv.org/abs/2101.06448)

在推荐系统中，当 user 与 item 的交互数据稀疏时，社交关系常常用来提高推荐质量。大多数现有的社会推荐模型 (Social Recommendation)
利用配对关系来挖掘潜在的用户偏好。然而，现实生活中的用户交互是非常复杂的，用户关系可能是高阶的。超图(Hypergraph) 为复杂的高阶关系提供了一种自然的建模方式，
而它在改善社会推荐方面的潜力却没有被充分发掘。在本文中，我们填补了这一空白，并提出了一个多通道超图卷积网络，通过利用高阶用户关系来提高社交推荐。从技术上讲，网络中的每个通道都编码了一个超图，通过超图卷积网络描述了一个共同的高阶用户关系图。通过聚合多个通道学习的嵌入向量，可以获得全面的用户表示，
从而产生推荐结果。然而，聚合操作也可能掩盖了不同类型的高阶连接信息的固有特征。为了弥补聚合的损失，创新性地将自我监督学习整合到超图聚合网络的训练中，以分层互信息最大化的方式重新获得连接信息。

![](https://tva1.sinaimg.cn/large/008i3skNly1gya578zf58j30tn078dgo.jpg)

![](https://tva1.sinaimg.cn/large/008i3skNly1gya57zcfs9j30v10c1q59.jpg)

以上两图分别展示了超图的组成类型及 MHCN 的网络结构。

```text
@inproceedings{yu2021self,
  title={Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation},
  author={Yu, Junliang and Yin, Hongzhi and Li, Jundong and Wang, Qinyong and Hung, Nguyen Quoc Viet and Zhang, Xiangliang},
  booktitle={Proceedings of the Web Conference 2021},
  pages={413--424},
  year={2021}
}
```

## 数据准备

论文使用了 3 个常见的公开的社交推荐数据集，`LastFM`、`Douban`、`Yelp`, 这里以 [LastFM](http://files.grouplens.org/datasets/hetrec2011/)
数据为例进行复现。该数据集包含文件如下：

```text
   * artists.dat
   
        This file contains information about music artists listened and tagged by the users.
   
   * tags.dat
   
   	This file contains the set of tags available in the dataset.

   * user_artists.dat
   
        This file contains the artists listened by each user.
        
        It also provides a listening count for each [user, artist] pair.

   * user_taggedartists.dat - user_taggedartists-timestamps.dat
   
        These files contain the tag assignments of artists provided by each particular user.
        
        They also contain the timestamps when the tag assignments were done.
   
   * user_friends.dat
   
   	These files contain the friend relations between users in the database.
```

MHCN 在 github 上开源了其处理好的数据集，包括 `ratings.txt` 和 `trusts.txt`
两个文件，下载地址: https://github.com/Coder-Yu/QRec/blob/master/dataset/lastfm 。

## 运行环境

PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos

## 快速开始

本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在 mhcn 模型目录的快速执行命令如下：

```bash
# 进入模型目录
# cd models/recall/mhcn
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 动态图预测 (注意：这里 mhcn 评估指标较为复杂，没有集成到通用 tools/infer.py 中）
python infer.py -m config.yaml
``` 

## 模型组网

MHCN 网络结构见上图 2 所示，与 net.py 文件中代码一一对应。

## 效果复现

![](https://tva1.sinaimg.cn/large/008i3skNly1gya5pggeiaj30nq02mt97.jpg)

原论文 MHCN 在 LastFM 数据集上 P@10、R@10、N@10 最佳结果分别为
20.052%、20.375%、0.24395，其 [github 开源代码](https://github.com/Coder-Yu/QRec/issues/216) issues 有反馈前两项指标稍低, NDCG@10 要稍高一些，
复现结果为 19.607%、19.914%、0.24459，但结果相对其他模型仍是最优的。

为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现 readme 中的效果,请按如下步骤依次操作即可。 在全量数据下模型的指标如下：  
| 模型 | P@10 | R@10 | N@10 | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------ | :------ | :------| :------ | 
| MHCN | 20.063% | 20.452% | 0.24780 | 2000 | 120 | 约 15 分钟 |

1. 确认您当前所在目录为 PaddleRec/models/recall/mhcn
2. 进入 PaddleRec/datasets/LastFM_MHCN 目录下，执行该脚本，会从国内源的服务器上下载数据集，并解压到指定文件夹。
```shell
cd ../../../datasets/LastFM_MHCN
sh run.sh
```
3. 切回模型目录,执行命令运行全量数据

```bash
# 需要在 mhcn 目录下
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml # 全量数据运行 config_bigdata.yaml 
python -u infer.py -m config_bigdata.yaml # 全量数据运行 config_bigdata.yaml 
```

## 进阶使用

## FAQ
