# 基于DIEN模型的点击率预估模型

**[AI Studio在线运行环境](https://aistudio.baidu.com/aistudio/projectdetail/3240212)**

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── train_data
        |—— sample_data #样例数据

├── __init__.py
├── README.md #文档
├── config.yaml # sample数据配置
├── config_bigdata.yaml # 全量数据配置
├── net.py # 模型核心组网（动静统一）
├── din_reader.py #数据读取程序
├── dygraph_model.py # 构建动态图
|── static_model.py # 构建静态图
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
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。本模型实现了下述论文中提出的rank模型：

```text
@inproceedings{
  title={Deep Interest Evolution Network for Click-Through Rate Prediction},
  author={Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhou, Kun Gai},
  year={2019}
}
```

DIEN模型引入GRU-Attention兴趣学习机制，设计局部激活单元，刻画用户兴趣。
从用户关于某个物品的历史行为数据中，学习用户的兴趣表达。
不同的商品/广告兴趣向量不同，从而提高模型的表达能力。

此外，本文提出小批量正则与数据自适应激活功能，
提高了工业级百亿级数据模型训练速度。

## 数据准备
此模型训练和预测涉及：用户历史点击商品序列、用户历史点击品类序列、推荐广告商品序列、推荐广告商品品类、点击标记；

每行的格式为：
用户历史点击商品序列、用户历史点击品类序列、推荐广告商品序列、推荐广告商品品类、点击标记
以上5项用分号分割；
用户历史点击商品序列中，商品间用空格隔开；
用户历史点击品类序列中，品类间用空格隔开；

数据处理中，
对于序列数据，我们以最长序列长度为准，将其他序列长度补齐，方便数据对齐过程中做计算；
同时，采用mask矩阵，对于补齐的网格部分，初始化为-INF，从而在sigmoid后，使之失效为0；

在模型目录的data/train_data/sample_data.txt目录下为您准备了快速运行的示例数据

## 运行环境
PaddlePaddle>=2.0

python 3.5/3.6/3.7

os : windows/linux/macos 

注：建议通过编译方式安装最新develop分支，不然可能会出现最新属性找不到的错误，安装方式见[教程](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/compile/arm-compile.html)

## 快速开始
本文提供了样例数据可以供您快速体验，在dien模型目录的快速执行命令如下： 
```bash
# 进入模型目录
cd models/rank/dien 
# 动态图训练
python -u ../../../tools/trainer.py -m config.yaml 
# 动态图预测
python -u ../../../tools/infer.py -m config.yaml 

# 静态图训练
python -u ../../../tools/static_trainer.py -m config.yaml 
# 静态图预测
python -u ../../../tools/static_infer.py -m config.yaml 
```
其中yaml文件的超参数解释如下：
```
item_emb_size: 商品的embedding维度
cat_emb_size:  品类的embedding维度
item_count: 商品的种类数目
cat_count:  品类的种类数目
```

#### Loss及Acc计算
- 预测的结果为一个sigmoid向量，表示推荐的商品广告被用户点击的概率
- 样本的损失函数值由CTR_loss和Auxiliary_loss叠加组成，同时计算预测的auc

## 效果复现
注：目前paddle未支持AUGRU算子，此处用的是普通的GRU，不同于论文部分。
### 数据集获取及预处理
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。同时，我们提供了全量数据生成的脚本，将会自动下载amazon eletronics dataset全量数据集并转换为模型能接受的
输入格式。

- 处理好的原始数据集下载执行方法：

```bash
cd ./PaddleRec/datasets/amazonElec_Din
sh run.sh
```

- 或自行进行原始数据下载预处理，执行方法如下：
```bash
cd ./PaddleRec/datasets/amazonElec_Din
sh data_process.sh
python build_dataset.py
# 脚本运行完成后，打开config.txt，将其中的商品的种类数目（第二行数值）、品类的种类数目信息（第三行数值），
# copy到config_bigdata.yaml里，替换超参数item_count cat_count
```

### 模型训练及效果复现
在全量数据下模型的指标如下： 
| 模型 | auc | batch_size | epoch_num| Time of each epoch| 
| :------| :------ | :------ | :------| :------ | 
| DIEN | 0.826 | 32 | 10 | 约2小时 | 

```bash
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml
# 静态图训练
python -u ../../../tools/static_trainer.py -m config_bigdata.yaml
# 动态图预测
python -u ../../../tools/infer.py -m config_bigdata.yaml
# 静态图预测
python -u ../../../tools/static_infer.py -m config_bigdata.yaml
```
