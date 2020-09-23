# 基于deepFM模型的点击率预估模型

## 介绍
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。本模型实现了下述论文中的deepFM模型：

```text
@inproceedings{guo2017deepfm,
  title={DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  booktitle={the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={1725--1731},
  year={2017}
}
```
## 数据准备
### 数据来源
训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```bash
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。

### 数据预处理
数据预处理共包括两步：
- 将原始训练集按9:1划分为训练集和验证集
- 数值特征（连续特征）需进行归一化处理，但需要注意的是，对每一个特征```<integer feature i>```，归一化时用到的最大值并不是用全局最大值，而是取排序后95%位置处的特征值作为最大值，同时保留极值。

### 一键下载训练及测试数据
```bash
sh run.sh
```
进入models/rank/deepfm/data目录下，执行该脚本，会从国内源的服务器上下载Criteo数据集，并解压到指定文件夹，然后自动处理数据转化为可直接进行训练的格式。解压后全量训练数据放置于`./train_datal`，全量测试数据放置于`./test_data`，可以直接输入的训练数据放置于`./slot_train_datal`，可直接输入的测试数据放置于`./slot_test_datal`
执行该脚本的理想输出为：
```
download and extract starting...
Downloading dac.tar.gz
[==================================================] 100.00%
Uncompress dac.tar.gz
[==================================================] 100.00%
Downloading deepfm%2Ffeat_dict_10.pkl2
[==================================================] 100.00%
download and extract finished
preprocessing...
('generating feature dict', 0)
('generating feature dict', 0)
('generating feature dict', 0)
('generating feature dict', 0)
...
('generating feature dict', 1)
('generating feature dict', 1)
('args.num_feat ', 1086460)
Done!
preprocess done
done
```

## 数据读取
为了能高速运行CTR模型的训练，`PaddleRec`封装了`dataset`与`dataloader`API进行高性能的数据读取。

如何在我们的训练中引入dataset读取方式呢？无需变更数据格式，只需在我们的训练代码中加入以下内容，便可达到媲美二进制读取的高效率，以下是一个比较完整的流程：

### 引入dataset

1. 通过工厂类`fluid.DatasetFactory()`创建一个dataset对象。
2. 将我们定义好的数据输入格式传给dataset，通过`dataset.set_use_var(inputs)`实现。
3. 指定我们的数据读取方式，由`dataset_generator.py`实现数据读取的规则，后面将会介绍读取规则的实现。
4. 指定数据读取的batch_size。
5. 指定数据读取的线程数，该线程数和训练线程应保持一致，两者为耦合的关系。
6. 指定dataset读取的训练文件的列表。

```python
def get_dataset(inputs, args)
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python dataset_generator.py")
    dataset.set_batch_size(args.batch_size)
    dataset.set_thread(int(args.cpu_num))
    file_list = [
        str(args.train_files_path) + "/%s" % x
        for x in os.listdir(args.train_files_path)
    ]
    logger.info("file list: {}".format(file_list))
    return dataset, file_list
```

### 如何指定数据读取规则

在上文我们提到了由`dataset_generator.py`实现具体的数据读取规则，那么，怎样为dataset创建数据读取的规则呢？
以下是`dataset_generator.py`的全部代码，具体流程如下：
1. 首先我们需要引入dataset的库，位于`paddle.fluid.incubate.data_generator`。
2. 声明一些在数据读取中会用到的变量，如示例代码中的`cont_min_`、`categorical_range_`等。
3. 创建一个子类，继承dataset的基类，基类有多种选择，如果是多种数据类型混合，并且需要转化为数值进行预处理的，建议使用`MultiSlotDataGenerator`；若已经完成了预处理并保存为数据文件，可以直接以`string`的方式进行读取，使用`MultiSlotStringDataGenerator`，能够进一步加速。在示例代码，我们继承并实现了名为`CriteoDataset`的dataset子类，使用`MultiSlotDataGenerator`方法。
4. 继承并实现基类中的`generate_sample`函数，逐行读取数据。该函数应返回一个可以迭代的reader方法(带有yield的函数不再是一个普通的函数，而是一个生成器generator，成为了可以迭代的对象，等价于一个数组、链表、文件、字符串etc.)
5. 在这个可以迭代的函数中，如示例代码中的`def reader()`，我们定义数据读取的逻辑。例如对以行为单位的数据进行截取，转换及预处理。
6. 最后，我们需要将数据整理为特定的格式，才能够被dataset正确读取，并灌入的训练的网络中。简单来说，数据的输出顺序与我们在网络中创建的`inputs`必须是严格一一对应的。在示例代码中，我们将数据整理成`click:value dense_feature:value ... dense_feature:value 1:value ... 26:value`的格式。用print输出是因为我们在run.sh中将结果重定向到slot_train_data等文件中，由模型直接读取。在用户自定义使用时，可以使用`zip`的方法将参数名与数值构成的元组组成了一个list，并将其yield输出，并在config.yaml中的data_converter参数指定reader的路径。

```python
class Reader(dg.MultiSlotDataGenerator):
    def __init__(self, config):
        dg.MultiSlotDataGenerator.__init__(self)
        _config = envs.load_yaml(config)
    def init(self):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [
            5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 46,
            231, 4008, 7393
        ]
        self.cont_diff_ = [
            self.cont_max_[i] - self.cont_min_[i]
            for i in range(len(self.cont_min_))
        ]
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)
        # load preprocessed feature dict
        self.feat_dict_name = "sample_data/feat_dict_10.pkl2"
        self.feat_dict_ = pickle.load(open(self.feat_dict_name, 'rb'))
    def _process_line(self, line):
        features = line.rstrip('\n').split('\t')
        feat_idx = []
        feat_value = []
        for idx in self.continuous_range_:
            if features[idx] == '':
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(self.feat_dict_[idx])
                feat_value.append(
                    (float(features[idx]) - self.cont_min_[idx - 1]) /
                    self.cont_diff_[idx - 1])
        for idx in self.categorical_range_:
            if features[idx] == '' or features[idx] not in self.feat_dict_:
                feat_idx.append(0)
                feat_value.append(0.0)
            else:
                feat_idx.append(self.feat_dict_[features[idx]])
                feat_value.append(1.0)
        label = [int(features[0])]
        return feat_idx, feat_value, label
    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """
        def data_iter():
            feat_idx, feat_value, label = self._process_line(line)
            s = ""
            for i in [('feat_idx', feat_idx), ('feat_value', feat_value),
                      ('label', label)]:
                k = i[0]
                v = i[1]
                for j in v:
                    s += " " + k + ":" + str(j)
            print(s.strip()) # add print for data preprocessing
        return data_iter
reader = Reader(
    "../config.yaml")  # run this file in original folder to find config.yaml
reader.init()
reader.run_from_stdin()
```
### 快速调试Dataset
我们可以脱离组网架构，单独验证Dataset的输出是否符合我们预期。使用命令
`cat 数据文件 | python dataset读取python文件`进行dataset代码的调试：
```bash
cat train_data/part-0 | python python get_slot_data.py
```
输出的数据格式如下：
`label:value dense_input:value ... dense_input:value sparse_input:value ... sparse_input:value `

理想的输出为(截取了一个片段)：
```bash
...
click:0 dense_feature:0.05 dense_feature:0.00663349917081 dense_feature:0.05 dense_feature:0.0 dense_feature:0.02159375 dense_feature:0.008 dense_feature:0.15 dense_feature:0.04 dense_feature:0.362 dense_feature:0.1 dense_feature:0.2 dense_feature:0.0 dense_feature:0.04 1:715353 2:817085 3:851010 4:833725 5:286835 6:948614 7:881652 8:507110 9:27346 10:646986 11:643076 12:200960 13:18464 14:202774 15:532679 16:729573 17:342789 18:562805 19:880474 20:984402 21:666449 22:26235 23:700326 24:452909 25:884722 26:787527
...
```

### deepFM模型组网

deepFM模型的组网本质是一个二分类任务，代码参考`model.py`。模型主要组成是一阶项部分，二阶项部分,dnn部分以及相应的分类任务的loss计算和auc计算。模型的组网可以看做FM部分和dnn部分的结合，其中FM部分主要的工作是通过特征间交叉得到低阶特征，以二阶特征为主。FM的表达式如下，可观察到，只是在线性表达式后面加入了新的交叉项特征及对应的权值。
$$Out=sigmoid(b + \sum^{N}_{i=1}W_iX_i + \sum^{N-1}_{i=1}\sum^{N}_{j=i+1}W_{ij}X_iX_j)$$

#### 一阶项部分
一阶项部分类似于我们rank下的logistic_regression模型。主要由embedding层和reduce_sum层组成  
首先介绍Embedding层的搭建方式：`Embedding`层的输入是`feat_idx`，shape由超参的`sparse_feature_number`定义。需要特别解释的是`is_sparse`参数，当我们指定`is_sprase=True`后，计算图会将该参数视为稀疏参数，反向更新以及分布式通信时，都以稀疏的方式进行，会极大的提升运行效率，同时保证效果一致。  
各个稀疏的输入通过Embedding层后，进行reshape操作，方便和连续值进行结合。  
将离散数据通过embedding查表得到的值，与连续数据的输入进行相乘再累加的操作，合为一个一阶项的整体。  
用公式表示如下：  
$$\sum^{N}_{i=1}W_iX_i$$

```python
first_weights_re = fluid.embedding(
    input=feat_idx,
    is_sparse=True,
    is_distributed=is_distributed,
    dtype='float32',
    size=[self.sparse_feature_number + 1, 1],
    padding_idx=0,
    param_attr=fluid.ParamAttr(
        initializer=fluid.initializer.TruncatedNormalInitializer(
            loc=0.0, scale=init_value_),
        regularizer=fluid.regularizer.L1DecayRegularizer(self.reg)))
first_weights = fluid.layers.reshape(first_weights_re,shape=[-1, self.num_field, 1])  # None * num_field * 1
y_first_order = fluid.layers.reduce_sum((first_weights * feat_value),1)
```

#### 二阶项部分
二阶项部分主要实现了公式中的交叉项部分，也就是特征的组合部分。Wij求解的思路是通过矩阵分解的方法。所有的二次项参数Wij可以组成一个对称阵W，那么这个矩阵就可以分解为 $W=V^TV$，V 的第 i 列便是第 i 维特征的隐向量。交叉项的展开式如下：
$$\sum^{N-1}_{i=1}\sum^{N}_{j=i+1}W_{ij}X_iX_j =1/2\sum^{k}_{j=1}((\sum^{N}_{i=1}W_iX_i)^2-\sum^{N}_{i=1}W_i^2X_i^2)$$


```python
feat_embeddings_re = fluid.embedding(
    input=feat_idx,
    is_sparse=True,
    is_distributed=is_distributed,
    dtype='float32',
    size=[self.sparse_feature_number + 1, self.sparse_feature_dim],
    padding_idx=0,
    param_attr=fluid.ParamAttr(
        initializer=fluid.initializer.TruncatedNormalInitializer(
            loc=0.0,
            scale=init_value_ /
            math.sqrt(float(self.sparse_feature_dim)))))
feat_embeddings = fluid.layers.reshape(
    feat_embeddings_re,
    shape=[-1, self.num_field, self.sparse_feature_dim])  # None * num_field * embedding_size
# None * num_field * embedding_size
feat_embeddings = feat_embeddings * feat_value
 # sum_square part
summed_features_emb = fluid.layers.reduce_sum(feat_embeddings, 1)  # None * embedding_size
summed_features_emb_square = fluid.layers.square(summed_features_emb)  # None * embedding_size
 # square_sum part
squared_features_emb = fluid.layers.square(feat_embeddings)  # None * num_field * embedding_size
squared_sum_features_emb = fluid.layers.reduce_sum(squared_features_emb, 1)  # None * embedding_size
y_second_order = 0.5 * fluid.layers.reduce_sum(summed_features_emb_square - squared_sum_features_emb,1,keep_dim=True)  # None * 1
```

#### dnn部分
相比fm模型，我们去除了fm模型中的偏移量，而加入了dnn部分作为特征间的高阶组合，通过并行的方式组合fm和dnn两种方法，两者共用底层的embedding数据。dnn部分的主要组成为三个全连接层，每层FC的输出维度都为400，每层FC都后接一个relu激活函数，每层FC的初始化方式为符合正态分布的随机初始化.  
最后接了一层输出维度为1的fc层，方便与fm部分综合计算预测值。  
```python
y_dnn = fluid.layers.reshape(feat_embeddings, [-1, self.num_field * self.sparse_feature_dim])
for s in self.layer_sizes:
    y_dnn = fluid.layers.fc(
        input=y_dnn,
        size=s,
        act=self.act,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormalInitializer(
                loc=0.0, scale=init_value_ / math.sqrt(float(10)))),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormalInitializer(
                loc=0.0, scale=init_value_)))
y_dnn = fluid.layers.fc(
    input=y_dnn,
    size=1,
    act=None,
    param_attr=fluid.ParamAttr(
        initializer=fluid.initializer.TruncatedNormalInitializer(
            loc=0.0, scale=init_value_)),
    bias_attr=fluid.ParamAttr(
        initializer=fluid.initializer.TruncatedNormalInitializer(
            loc=0.0, scale=init_value_)))

```

#### Loss及Auc计算
- 预测的结果将FM的一阶项部分，二阶项部分以及dnn部分相加，再通过激活函数sigmoid给出，为了得到每条样本分属于正负样本的概率，我们将预测结果和`1-predict`合并起来得到predict_2d，以便接下来计算auc。  
- 每条样本的损失为负对数损失值，label的数据类型将转化为float输入。  
- 该batch的损失`avg_cost`是各条样本的损失之和
- 我们同时还会计算预测的auc，auc的结果由`fluid.layers.auc()`给出，该层的返回值有三个，分别是全局auc: `auc_var`，当前batch的auc: `batch_auc_var`，以及auc_states: `_`，auc_states包含了`batch_stat_pos, batch_stat_neg, stat_pos, stat_neg`信息。
```python
self.predict = fluid.layers.sigmoid(y_first_order + y_second_order +y_dnn)
cost = fluid.layers.log_loss(
    input=self.predict, label=fluid.layers.cast(self.label, "float32"))
avg_cost = fluid.layers.reduce_sum(cost)
self._cost = avg_cost
predict_2d = fluid.layers.concat([1 - self.predict, self.predict], 1)
label_int = fluid.layers.cast(self.label, 'int64')
auc_var, batch_auc_var, _ = fluid.layers.auc(input=predict_2d,
                                                     label=label_int,
                                                     slide_steps=0)
```

完成上述组网后，我们最终可以通过训练拿到`auc`指标。
```
PaddleRec: Runner infer_runner Begin
Executor Mode: infer
processor_register begin
Running SingleInstance.
Running SingleNetwork.
Warning:please make sure there are no hidden files in the dataset folder and check these hidden files:[]
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from incerement/1
2020-09-23 11:26:38,879-INFO:   [Infer] batch: 1, time_each_interval: 1.39s, AUC: [0.8044914]
2020-09-23 11:26:39,468-INFO:   [Infer] batch: 2, time_each_interval: 0.59s, AUC: [0.80449145]
2020-09-23 11:26:40,021-INFO:   [Infer] batch: 3, time_each_interval: 0.55s, AUC: [0.80449146]
2020-09-23 11:26:40,557-INFO:   [Infer] batch: 4, time_each_interval: 0.54s, AUC: [0.80449131]
2020-09-23 11:26:41,148-INFO:   [Infer] batch: 5, time_each_interval: 0.59s, AUC: [0.80449144]
2020-09-23 11:26:41,659-INFO:   [Infer] batch: 6, time_each_interval: 0.51s, AUC: [0.80449172]
2020-09-23 11:26:42,182-INFO:   [Infer] batch: 7, time_each_interval: 0.52s, AUC: [0.80449169]
2020-09-23 11:26:42,691-INFO:   [Infer] batch: 8, time_each_interval: 0.51s, AUC: [0.80449169]
2020-09-23 11:26:43,190-INFO:   [Infer] batch: 9, time_each_interval: 0.50s, AUC: [0.80449191]
2020-09-23 11:26:43,671-INFO:   [Infer] batch: 10, time_each_interval: 0.48s, AUC: [0.80449164]
...
2020-09-23 12:16:10,279-INFO:   [Infer] batch: 8982, time_each_interval: 0.18s, AUC: [0.8044914]
2020-09-23 12:16:10,518-INFO:   [Infer] batch: 8983, time_each_interval: 0.24s, AUC: [0.80449133]
Infer infer_phase of epoch 1 done, use time: 1764.81796193, global metrics: AUC=0.80449133
PaddleRec Finish
```
