# BERT4Rec_AC
飞浆论文复现挑战赛第四期 BERT4Rec

原论文地址 https://dl.acm.org/doi/abs/10.1145/3357384.3357895

原论文代码 https://github.com/FeiSun/BERT4Rec

参考实现 https://github.com/Qdriving/Bert4Rec_Paddle2.0

# 模型描述
BERT4Rec将NLP中的mask language任务迁移到序列推荐问题来，**给予了序列推荐一种不同于item2item，left2right的训练范式**。

具体来说，对于一条物品序列，**以一定的概率p随机mask掉序列中的物品**，**使用transformer的encoder结构**对mask item进行预测。

通过数据增强，完形填空任务的方式使得训练更加充分。

# 复现精度

BERT4Rec论文的一个创新点是将nlp领域完形填空式的任务引入序列推荐 具体就体现在对序列数据的增强上

我们根据原论文和作者开源代码的实现 对不同数据集设置数据增强的参数：mask proportion 0.6 for beauty, 0.2 for ML-1m. Dual factor = 10 

模型参数设置上根据论文作者提供的json文件进行设置，最终复现效果如下：



| Data | Hit@10 | NDCG@10 | MRR|
|:-------:|:-----|--------:|--------|
|Beauty| 0.312301 | 0.190869 |0.172197 |
|ML-1m| 0.692595| 0.486158 | 0.432563|


# 环境依赖
- 硬件：CPU、GPU
- 框架： 
   - PaddlePaddle >= 2.0.0 
   - Python >= 3.6
        

# 数据生成与数据增强

Beauty: http://jmcauley.ucsd.edu/data/amazon/

MovieLens: https://grouplens.org/datasets/movielens

**下载数据放置到 ./bert_train/data/**


# 快速开始

**1.数据增强**

运行./bert_train/gen_data_ml1m.py进行ML-1m的数据生成与增强

运行./bert_train/gen_data_beauty.py进行ML-1m的数据生成与增强

**2.候选集采样与生成**

这里需要说明的是，在序列推荐，包括BERT4Rec，为了降低inference的时间，对于每一个target item会采样100个负样本。即是说将候选物品的数量限制在101个（**一个正样本，100个负样本**）

我们根据作者开源代码进行负样本的采样。（**根据流行度采样**）

运行./candidate_gen.py 进行候ML-1m数据集候选集的生成

运行./candidate_gen_beauty.py 进行候beauty数据集候选集的生成

**3.训练与预测**

- **到根目录下运行以下语句以复现ML-1m数据集上的效果**
```
CUDA_VISIBLE_DEVICES=0 python3 ml1m.py 
```


- **到根目录下运行以下语句以复现Beauty数据集上的效果**
```
CUDA_VISIBLE_DEVICES=1 python3 beauty.py 
```

- Fell free to change the parameter setting.

# 代码结构与详细说明

```
|--bert4rec
   |--bert4rec_ac.py          # BERT4Rec模型文件
   |--modules.py              # 组块文件
   |--dataset.py              # dataset加载类            
|--bert_train                     
   |--data                        # 存放数据
   |--....                        # 模型config，数据增强函数
|--utils                          # 工具类
|--evaluate.py                    # 工具类-评估函数
|--beauty.py                      # 复现Beauty数据集
|--ml1m.py                        # 复现ML-1m数据集
|--README.md                      # readme
|--candidate_gen.py               # 候选集采样与生成-ML-1m
|--candidate_gen_beauty.py        # 候选集采样与生成-Beauty
|--beauty_log.txt                 # 复现Beauty的日志
|--ml1m_log.txt                   # 复现ML-1m的日志1
|--ml1m_log2.txt                  # 复现ML-1m的日志2
|--ml1m_log3.txt                  # 复现ML-1m的日志3
```
