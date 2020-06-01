# DSSM

## 简介

DSSM[《Learning Deep Structured Semantic Models for Web Search using Clickthrough Data》]( https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf )即基于深度网络的语义模型，其核心思想是将query和doc映射到共同维度的语义空间中，通过最大化query和doc语义向量之间的余弦相似度，从而训练得到隐含语义模型，达到检索的目的，并通过word hashing方法来减少输入向量的维度。DSSM有很广泛的应用，比如：搜索引擎检索，广告相关性，问答系统，机器翻译等。

本项目按照论文的网络结构在paddlepaddle上实现DSSM模型，并构造数据集验证网络的正确性。

## 模型超参
```
optimizer:
  class: sgd                          # 优化器
  learning_rate: 0.01                 # 学习率
  strategy: async                     # 参数更新方式
TRIGRAM_D: 1000                       # query和doc语义向量长度
NEG: 4                                # 负采样个数 
fc_sizes: [300, 300, 128]             # fc层大小
fc_acts: ['tanh', 'tanh', 'tanh']     # fc层激活函数

```

## 快速开始
PaddleRec内置了demo小数据，方便用户快速使用模型，训练命令如下：
```bash
python -m paddlerec.run -m paddlerec.models.match.dssm
```

执行预测前，需更改config.yaml中的配置，具体改动如下：
```
workspace: "~/code/paddlerec/models/match/dssm"     # 改为当前config.yaml所在的绝对路径

#mode: runner1     # train
mode: runner2     # infer

runner:
- name: runner2
  class: single_infer
  init_model_path: "increment/2"       # 改为需要预测的模型路径

phase:
- name: phase1
  model: "{workspace}/model.py"
  dataset_name: dataset_infer          # 改成预测dataset
  thread_num: 1                        # dataset线程数
```
改完之后，执行预测命令：
```
python -m paddlerec.run -m ./config.yaml
```

## 提测说明
当前，DSSM模型采用的数据集是随机构造的，因此提测仅需按上述步骤在demo数据集上跑通即可。
