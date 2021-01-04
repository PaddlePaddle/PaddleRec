# 基于GATE DNN模型的点击率预估模型

## 介绍
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。本模型实现了下述论文中提出的DNN模型：

```text
@inproceedings{
  title={GateNet: Gating-Enhanced Deep Network for Click-Through Rate Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  year={2020}
}
```
其中的embedding_gate实现采用了论文中默认的private field和vec-wise model.
embedding_gate和hidden_gate的开关分别对应yaml配置文件内（如config.yaml)hyper_parameters下的use_embedding_gate和use_hidden_gate变量，
默认情况下2个开关均处于打开状态，若需要关闭某个开关，只需要把它的值设置成False并保存配置文件即可。当2个开关都关闭，运行的算法即为标准的dnn。
#
## 数据准备
### 数据来源及预处理
可参考DNN模型readme'数据来源及预处理'部分
### 一键下载训练及测试数据
```bash
sh run.sh
```
进入models/rank/gateDnn/data目录下，执行该脚本，会从国内源的服务器上下载Criteo数据集，并解压到指定文件夹。
原始的全量数据放置于`./train_data_full/`，原始的全量测试数据放置于`./test_data_full/`，原始的用于快速验证的训练数据与测试数据放置于`./train_data/`与`./test_data/`。处理后的全量训练数据放置于`./slot_train_data_full/`，处理后的全量测试数据放置于`./slot_test_data_full/`，处理后的用于快速验证的训练数据与测试数据放置于`./slot_train_data/`与`./slot_test_data/`。
至此，我们已完成数据准备的全部工作。数据读取方式可以参考DNN模型readme部分

## 模型组网
### 数据输入声明
正如数据准备章节所介绍，Criteo数据集中，分为连续数据与离散（稀疏）数据，所以整体而言，GATE-DNN模型的数据输入层包括三个，分别是：`dense_input`用于输入连续数据，维度由超参数`dense_input_dim`指定，数据类型是归一化后的浮点型数据。`sparse_inputs`用于记录离散数据，在Criteo数据集中，共有26个slot，所以我们创建了名为`1~26`的26个稀疏参数输入，数据类型为整数；最后是每条样本的`label`，代表了是否被点击，数据类型是整数，0代表负样例，1代表正样例。

### CTR-DNN模型组网

GATE-DNN模型主要组成是一个`Embedding`层,三个`FC`层，以及相应的分类任务的loss计算和auc计算。

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

### 效果复现
在全量数据下模型的指标如下：
| 模型 | auc | batch_size | thread_num| epoch_num| Time of each epoch| GateType|

| gateDnn | 0.7974 | 512 | 1 | 4 | 约5小时 | embeddingGate + hiddenGate |

| dnn      | 0.7959 | 512 | 1 | 4 | 约2小时 | None                       |

1. 确认您当前所在目录为PaddleRec/models/rank/gateDnn  
2. 在data目录下运行数据一键处理脚本，处理时间较长，请耐心等待。命令如下：  
``` 
cd data
sh run.sh
cd ..
```
3. 修改config_bigdata.yaml文件，决定是否使用embedding-gate和hidden-gate。若不能使用GPU运行，需要把dygraph里use_gpu变量改成False
4. 运行命令，模型会进行四个epoch的训练，然后预测第四个epoch，并获得相应auc指标  
```
python -u train.py -m config_bigdata.yaml
```
5. 经过全量数据训练后，执行预测
```
python -u infer.py -m config_bigdata.yaml
```
