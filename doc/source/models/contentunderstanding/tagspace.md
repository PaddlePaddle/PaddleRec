# tagspace (TagSpace: Semantic Embeddings from Hashtags)

代码请参考：[tagspace文本分类模型](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/contentunderstanding/tagspace)  
如果我们的代码对您有用，还请点个star啊~  
关注我们，多少你能懂一点  

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
tagspace模型是一种对文本打标签的方法，来自论文论文[TAGSPACE: Semantic Embeddings from Hashtags](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.668.2094&rep=rep1&type=pdf)，它主要学习从短文到相关主题标签的映射。论文中主要利用CNN做doc向量， 然后优化 f(w,t+),f(w,t-)的距离作为目标函数，得到了 t（标签）和doc在一个特征空间的向量表达，这样就可以找 doc的hashtags了。  

## 数据准备
本模型使用论文中的ag_news数据集，在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分，数据的格式如下：  
```
2,27 7062 8390 456 407 8 11589 3166 4 7278 31046 33 3898 2897 426 1
2,27 9493 836 355 20871 300 81 19 3 4125 9 449 462 13832 6 16570 1380 2874 5 0 797 236 19 3688 2106 14 8615 7 209 304 4 0 123 1
2,27 12754 637 106 3839 1532 66 0 379 6 0 1246 9 307 33 161 2 8100 36 0 350 123 101 74 181 0 6657 4 0 1222 17195 1
```

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 


## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在tagspace模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/contentunderstanding/tagspace # 在任意目录均可运行
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
论文[TAGSPACE: Semantic Embeddings from Hashtags](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.668.2094&rep=rep1&type=pdf)中的网络结构如图所示，一层输入层，一个卷积层，一个pooling层以及最后一个全连接层进行降维。
<p align="center">
<img align="center" src="../../../doc/imgs/tagspace.png">
<p>


## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
在全量数据下模型的指标如下：  

| 模型 | acc | batch_size | epoch_num| Time of each epoch |
| :------| :------ | :------ | :------| :------ | 
| tagspace | 0.97 | 128 | 1 | 约7分钟 |

1. 确认您当前所在目录为PaddleRec/models/contentunderstanding/tagspace  
2. 进入paddlerec/datasets/ag_news目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的ag_news全量数据集，并解压到指定文件夹。
``` bash
cd ../../../datasets/ag_news
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
