## 数据准备
训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。  
在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分。

1. 确认您当前所在目录为PaddleRec/models/rank/ffm
2. 进入paddlerec/datasets/criteo目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的criteo全量数据集，并解压到指定文件夹。
``` bash
cd ../../../datasets/criteo
sh run.sh

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在ffm模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/rank/ffm # 在任意目录均可运行
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml # 样例数据运行config.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config_bigdata.yaml 

# 静态图训练
python -u ../../../tools/static_trainer.py -m config_bigdata.yaml # 样列数据运行config.yaml 
# 静态图预测
python -u ../../../tools/static_infer.py -m config_bigdata.yaml 
``` 



## 进阶使用
  
## FAQ
