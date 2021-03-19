# MIND(Multi-Interest Network with Dynamic Routing)

以下是本例的简要目录结构及说明： 
```shell
├── data #样例数据
│   ├── demo                    #demo训练数据
│   │   └── demo.txt     
│   ├── processs.py             #处理全量数据的脚本
│   ├── run.sh                  #全量数据下载的脚本
│   └── valid                    #demo测试数据
│       └── part-0    
├── config.yaml                 #数据配置
├── dygraph_model.py            #构建动态图
├── evaluate_dygraph.py         #评测动态图
├── evaluate_reader.py          #评测数据reader
├── evaluate_static.py          #评测静态图
├── mind_reader.py              #训练数据reader
├── net.py                      #模型核心组网（动静合一）
└── static_model.py             #构建静态图
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
本例实现了基于动态路由的用户多兴趣网络，如下图所示：
<p align="center">
<img align="center" src="../../../doc/imgs/mind.png">
<p>
Multi-Interest Network with Dynamic Routing (MIND) 是通过构建用户和商品向量在统一的向量空间的多个用户兴趣向量，以表达用户多样的兴趣分布。然后通过向量召回技术，利用这多个兴趣向量去检索出TopK个与其近邻的商品向量，得到 TopK个 用户感兴趣的商品。其核心是一个基于胶囊网络和动态路由的（B2I Dynamic Routing）Multi-Interest Extractor Layer。

推荐参考论文:[http://cn.arxiv.org/abs/1904.08030](http://cn.arxiv.org/abs/1904.08030)

## 数据准备
在模型目录的data目录下为您准备了快速运行的示例数据，训练数据、测试数据、词表文件依次保存在data/train, data/test文件夹中。若需要使用全量数据可以参考下方效果复现部分。

训练数据的格式如下：
```
0,17978,0
0,901,1
0,97224,2
0,774,3
0,85757,4
```
分别表示uid、item_id和点击的顺序(时间戳)

测试数据的格式如下：
```
user_id:543354 hist_item:143963 hist_item:157508 hist_item:105486 hist_item:40502 hist_item:167813 hist_item:233564 hist_item:221866 hist_item:280310 hist_item:61638 hist_item:158494 hist_item:74449 hist_item:283630 hist_item:135155 hist_item:96176 hist_item:20139 hist_item:89420 hist_item:247990 hist_item:126605 target_item:172183 target_item:114193 target_item:79966 target_item:134420 target_item:50557
user_id:543362 hist_item:119546 hist_item:78597 hist_item:86809 hist_item:63551 target_item:326165
user_id:543366 hist_item:45463 hist_item:9903 hist_item:3956 hist_item:49726 target_item:199426
```
其中`hist_item`和`target_item`均是变长序列，读取方式可以看`evaluate_reader.py`

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos

## 快速开始

在mind模型目录的快速执行命令如下：
```
# 进入模型目录
# cd models/recall/word2vec # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml 
# 动态图预测
python -u evaluate_dygraph.py -m config.yaml  -top_n 50  #对测试数据进行预测，并通过faiss召回候选结果评测Reacll、NDCG、HitRate指标

# 静态图训练
python -u ../../../tools/static_trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
# 静态图预测
python -u evaluate_static.py -m config.yaml  -top_n 50  #对测试数据进行预测，并通过faiss召回候选结果评测Reacll、NDCG、HitRate指标
```

## 模型组网

细节见上面[模型简介](#模型简介)部分

### 效果复现
由于原始论文没有提供实验的复现细节，为了方便使用者能够快速的跑通每一个模型，我们使用论文[ComiRec](https://arxiv.org/abs/2005.09347)提供的AmazonBook数据集和训练任务进行复现。我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。 

在全量数据下模型的指标如下：
| 模型 |  batch_size | epoch_num| Recall@50 | NDCG@50 | HitRate@50 |Time of each epoch |
| :------| :------ | :------ | :------| :------ | :------|  :------ | 
| mind(静态图) | 128 | 6 | 4.61% | 11.28%| 18.92%| -- |
| mind(动态图) | 128 | 6 | 4.57% | 11.25%| 18.99%| -- |

1. 确认您当前所在目录为PaddleRec/models/recall/mind
2. 进入data目录下执行run.sh脚本，会下载处理完成的AmazonBook数据集，并解压到指定目录
```bash
cd ./data
sh run.sh
``` 
3. 切回模型目录,执行命令运行全量数据
```bash
d - # 切回模型目录
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml # 全量数据运行config.yaml 
python -u evaluate_dygraph.py -m config.yaml # 全量数据运行config.yaml 
```

## 进阶使用
  
## FAQ

## 参考

数据集及训练任务参考了[ComiRec](https://github.com/THUDM/ComiRec)
