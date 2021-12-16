# 基于GateNet模型的点击率预估模型

**[AI Studio在线运行环境](https://aistudio.baidu.com/aistudio/projectdetail/3240375)**

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── sample_data #样例数据
        ├── train
            ├── sample_train.txt #训练数据样例
├── __init__.py
├── README.md #文档
├── config.yaml # sample数据配置
├── config_bigdata.yaml # 全量数据配置
├── net.py # 模型核心组网（动静统一）
├── agnews_reader.py #数据读取程序
├── static_model.py # 构建静态图
├── dygraph_model.py # 构建动态图
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
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。本模型实现了下述论文中提出的DNN模型：

```
@inproceedings{
  title={GateNet: Gating-Enhanced Deep Network for Click-Through Rate Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  year={2020}
}
```
其中的embedding_gate实现采用了论文中默认的private field和vec-wise model.
embedding_gate和hidden_gate的开关分别对应yaml配置文件内（如config.yaml)hyper_parameters下的use_embedding_gate和use_hidden_gate变量，
默认情况下2个开关均处于打开状态，若需要关闭某个开关，只需要把它的值设置成False并保存配置文件即可。当2个开关都关闭，运行的算法即为标准的dnn。

## 数据准备
可参考DNN模型readme'数据准备'部分，在模型目录的data目录下为您准备了快速运行的示例数据，若需要使用全量数据可以参考下方[效果复现](#效果复现)部分。数据的格式如下：  
```
click:0 dense_feature:0.0 dense_feature:0.00497512437811 dense_feature:0.05 dense_feature:0.08 dense_feature:0.207421875 dense_feature:0.028 dense_feature:0.35 dense_feature:0.08 dense_feature:0.082 dense_feature:0.0 dense_feature:0.4 dense_feature:0.0 dense_feature:0.08 1:737395 2:210498 3:903564 4:286224 5:286835 6:906818 7:906116 8:67180 9:27346 10:51086 11:142177 12:95024 13:157883 14:873363 15:600281 16:812592 17:228085 18:35900 19:880474 20:984402 21:100885 22:26235 23:410878 24:798162 25:499868 26:306163
click:1 dense_feature:0.0 dense_feature:0.932006633499 dense_feature:0.02 dense_feature:0.14 dense_feature:0.0395625 dense_feature:0.328 dense_feature:0.98 dense_feature:0.12 dense_feature:1.886 dense_feature:0.0 dense_feature:1.8 dense_feature:0.0 dense_feature:0.14 1:715353 2:761523 3:432904 4:892267 5:515218 6:948614 7:266726 8:67180 9:27346 10:266081 11:286126 12:789480 13:49621 14:255651 15:47663 16:79797 17:342789 18:616331 19:880474 20:984402 21:242209 22:26235 23:669531 24:26284 25:269955 26:187951
click:0 dense_feature:0.0 dense_feature:0.00829187396352 dense_feature:0.08 dense_feature:0.06 dense_feature:0.14125 dense_feature:0.076 dense_feature:0.05 dense_feature:0.22 dense_feature:0.208 dense_feature:0.0 dense_feature:0.2 dense_feature:0.0 dense_feature:0.06 1:737395 2:952384 3:511141 4:271077 5:286835 6:948614 7:903547 8:507110 9:27346 10:56047 11:612953 12:747707 13:977426 14:671506 15:158148 16:833738 17:342789 18:427155 19:880474 20:537425 21:916237 22:26235 23:468277 24:676936 25:751788 26:363967
```

## 运行环境
PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : windows/linux/macos 

## 快速开始
本文提供了样例数据可以供您快速体验，在任意目录下均可执行。在gatednn模型目录的快速执行命令如下： 
```bash
# 进入模型目录
# cd models/rank/gatenet # 在任意目录均可运行
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
### 数据输入声明
正如数据准备章节所介绍，Criteo数据集中，分为连续数据与离散（稀疏）数据，所以整体而言，Gate-Net模型的数据输入层包括三个，分别是：`dense_input`用于输入连续数据，维度由超参数`dense_input_dim`指定，数据类型是归一化后的浮点型数据。`sparse_inputs`用于记录离散数据，在Criteo数据集中，共有26个slot，所以我们创建了名为`1~26`的26个稀疏参数输入，数据类型为整数；最后是每条样本的`label`，代表了是否被点击，数据类型是整数，0代表负样例，1代表正样例。

### GateNet模型组网

GateNet模型主要组成是一个`Embedding`层,三个`FC`层，以及相应的分类任务的loss计算和auc计算。

#### Embedding层
首先介绍Embedding层的搭建方式：`Embedding`层的输入是`sparse_input`，由超参的`sparse_feature_number`和`sparse_feature_dim`定义。需要特别解释的是`is_sparse`参数，当我们指定`is_sprase=True`后，计算图会将该参数视为稀疏参数，反向更新以及分布式通信时，都以稀疏的方式进行，会极大的提升运行效率，同时保证效果一致。
```
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))
```

各个稀疏的输入通过Embedding层后，分别产生一个对应的embedding向量，若设置超参数use_embedding_gate=True， 则对应的embedding向量将通过一个embedding gate产生一个新embedding向量。所有embedding向量会被合并起来，置于一个list内，以方便进行concat的操作。
```
        if self.use_embedding_gate:
            self.embedding_gate_weight = [paddle.create_parameter(shape=[1], dtype="float32", name='embedding_gate_weight_%d' % i, default_initializer=paddle.nn.initializer.Normal(
                             std=1.0)) for i in range(num_field)]
                             
        if self.use_embedding_gate:
            for i in range(len(self.embedding_gate_weight)):
                emb = self.embedding(sparse_inputs[i])
                emb = paddle.reshape(
                    emb, shape=[-1, self.sparse_feature_dim
                                ])  # emb shape [batchSize, sparse_feature_dim]
                gate = paddle.sum(paddle.multiply(
                    emb, self.embedding_gate_weight[i]), axis=-1, keepdim=True)  # gate shape [batchSize,1]
                activate_gate = paddle.nn.functional.sigmoid(
                    gate)  # activate_gate [batchSize,1]
                emb = paddle.multiply(
                    emb, activate_gate)  # emb shape [batchSize, sparse_feature_dim]
                sparse_embs.append(emb)

```

#### FC层
将离散数据通过embedding查表得到的值，与连续数据的输入进行`concat`操作，合为一个整体输入，作为全链接层的原始输入。我们共设计了3层FC，每层FC的输出维度由超参`fc_sizes`指定，每层FC都后接一个`relu`激活函数.
```
        for i in range(len(layer_sizes)):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            act = paddle.nn.ReLU()
            self.add_sublayer('act_%d' % i, act)
            self._mlp_layers.append(act)

```
若设置超参数use_hidden_gate=True，则通过激活函数后的向量会继续通过一层hidden_gate产生一个新向量。每层FC的初始化方式为符合正态分布的随机初始化，标准差与上一层的输出维度的平方根成反比。

#### Loss及Auc计算
- 预测的结果通过一个输出shape为1的FC层给出，该FC层的激活函数是sigmoid，表示每条样本分属于正样本的概率。
- 样本的损失函数值由交叉熵给出
- 我们同时还会计算预测的auc

## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
在全量数据下模型的指标如下:  
| 模型 | auc | batch_size | epoch_num| Time of each epoch| GateType|  
| :------| :------ | :------ | :------| :------ | :------ |  
| gatenet | 0.7974 | 512 | 4 | 约5小时 | embeddingGate + hiddenGate |  
| dnn      | 0.7959 | 512 | 4 | 约2小时 | None                       |  

1. 确认您当前所在目录为PaddleRec/models/rank/gatenet 
2. 进入paddlerec/datasets/criteo目录下，执行该脚本，会从国内源的服务器上下载我们预处理完成的Criteo全量数据集，并解压到指定文件夹。
```bash
cd ../../../datasets/criteo
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
