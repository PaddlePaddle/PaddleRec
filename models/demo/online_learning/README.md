# 基于DNN模型的点击率预估模型

## 介绍
`CTR(Click Through Rate)`，即点击率，是“推荐系统/计算广告”等领域的重要指标，对其进行预估是商品推送/广告投放等决策的基础。简单来说，CTR预估对每次广告的点击情况做出预测，预测用户是点击还是不点击。CTR预估模型综合考虑各种因素、特征，在大量历史数据上训练，最终对商业决策提供帮助。本模型实现了下述论文中提出的DNN模型：

```text
@inproceedings{guo2017deepfm,
  title={DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author={Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li and Xiuqiang He},
  booktitle={the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI)},
  pages={1725--1731},
  year={2017}
}
```

#
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
sh download_data.sh
```
执行该脚本，会从国内源的服务器上下载Criteo数据集，并解压到指定文件夹。全量训练数据放置于`./train_data_full/`，全量测试数据放置于`./test_data_full/`，用于快速验证的训练数据与测试数据放置于`./train_data/`与`./test_data/`。

执行该脚本的理想输出为：
```bash
> sh download_data.sh
--2019-11-26 06:31:33--  https://fleet.bj.bcebos.com/ctr_data.tar.gz
Resolving fleet.bj.bcebos.com... 10.180.112.31
Connecting to fleet.bj.bcebos.com|10.180.112.31|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4041125592 (3.8G) [application/x-gzip]
Saving to: “ctr_data.tar.gz”

100%[==================================================================================================================>] 4,041,125,592  120M/s   in 32s

2019-11-26 06:32:05 (120 MB/s) - “ctr_data.tar.gz” saved [4041125592/4041125592]

raw_data/
raw_data/part-55
raw_data/part-113
...
test_data/part-227
test_data/part-222
Complete data download.
Full Train data stored in ./train_data_full
Full Test data stored in ./test_data_full
Rapid Verification train data stored in ./train_data
Rapid Verification test data stored in ./test_data
```
至此，我们已完成数据准备的全部工作。

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
6. 最后，我们需要将数据整理为特定的格式，才能够被dataset正确读取，并灌入的训练的网络中。简单来说，数据的输出顺序与我们在网络中创建的`inputs`必须是严格一一对应的，并转换为类似字典的形式。在示例代码中，我们使用`zip`的方法将参数名与数值构成的元组组成了一个list，并将其yield输出。如果展开来看，我们输出的数据形如`[('dense_feature',[value]),('C1',[value]),('C2',[value]),...,('C26',[value]),('label',[value])]`


```python
import paddle.fluid.incubate.data_generator as dg

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)

class CriteoDataset(dg.MultiSlotDataGenerator):
   
    def generate_sample(self, line):
        
        def reader():
            features = line.rstrip('\n').split('\t')
            dense_feature = []
            sparse_feature = []
            for idx in continuous_range_:
                if features[idx] == "":
                    dense_feature.append(0.0)
                else:
                    dense_feature.append(
                        (float(features[idx]) - cont_min_[idx - 1]) /
                        cont_diff_[idx - 1])
            for idx in categorical_range_:
                sparse_feature.append(
                    [hash(str(idx) + features[idx]) % hash_dim_])
            label = [int(features[0])]
            process_line = dense_feature, sparse_feature, label
            feature_name = ["dense_feature"]
            for idx in categorical_range_:
                feature_name.append("C" + str(idx - 13))
            feature_name.append("label")

            yield zip(feature_name, [dense_feature] + sparse_feature + [label])

        return reader

d = CriteoDataset()
d.run_from_stdin()
```
### 快速调试Dataset
我们可以脱离组网架构，单独验证Dataset的输出是否符合我们预期。使用命令
`cat 数据文件 | python dataset读取python文件`进行dataset代码的调试：
```bash
cat train_data/part-0 | python dataset_generator.py
```
输出的数据格式如下：
` dense_input:size ; dense_input:value ; sparse_input:size ; sparse_input:value ; ... ; sparse_input:size ; sparse_input:value ; label:size ; label:value `

理想的输出为(截取了一个片段)：
```bash
...
13 0.05 0.00663349917081 0.05 0.0 0.02159375 0.008 0.15 0.04 0.362 0.1 0.2 0.0 0.04 1 715353 1 817085 1 851010 1 833725 1 286835 1 948614 1 881652 1 507110 1 27346 1 646986 1 643076 1 200960 1 18464 1 202774 1 532679 1 729573 1 342789 1 562805 1 880474 1 984402 1 666449 1 26235 1 700326 1 452909 1 884722 1 787527 1 0
...
```

#
## 模型组网
### 数据输入声明
正如数据准备章节所介绍，Criteo数据集中，分为连续数据与离散（稀疏）数据，所以整体而言，CTR-DNN模型的数据输入层包括三个，分别是：`dense_input`用于输入连续数据，维度由超参数`dense_feature_dim`指定，数据类型是归一化后的浮点型数据。`sparse_input_ids`用于记录离散数据，在Criteo数据集中，共有26个slot，所以我们创建了名为`C1~C26`的26个稀疏参数输入，并设置`lod_level=1`，代表其为变长数据，数据类型为整数；最后是每条样本的`label`，代表了是否被点击，数据类型是整数，0代表负样例，1代表正样例。

在Paddle中数据输入的声明使用`paddle.fluid.data()`，会创建指定类型的占位符，数据IO会依据此定义进行数据的输入。
```python
dense_input = fluid.data(name="dense_input",
                                 shape=[-1, args.dense_feature_dim],
                                 dtype="float32")

sparse_input_ids = [
    fluid.data(name="C" + str(i),
                shape=[-1, 1],
                lod_level=1,
                dtype="int64") for i in range(1, 27)
]

label = fluid.data(name="label", shape=[-1, 1], dtype="int64")
inputs = [dense_input] + sparse_input_ids + [label]
```

### CTR-DNN模型组网

CTR-DNN模型的组网比较直观，本质是一个二分类任务，代码参考`model.py`。模型主要组成是一个`Embedding`层，三个`FC`层，以及相应的分类任务的loss计算和auc计算。

#### Embedding层
首先介绍Embedding层的搭建方式：`Embedding`层的输入是`sparse_input`，shape由超参的`sparse_feature_dim`和`embedding_size`定义。需要特别解释的是`is_sparse`参数，当我们指定`is_sprase=True`后，计算图会将该参数视为稀疏参数，反向更新以及分布式通信时，都以稀疏的方式进行，会极大的提升运行效率，同时保证效果一致。

各个稀疏的输入通过Embedding层后，将其合并起来，置于一个list内，以方便进行concat的操作。

```python
def embedding_layer(input):
   return fluid.layers.embedding(
            input=input,
            is_sparse=True,
            size=[args.sparse_feature_dim, 
                  args.embedding_size],
            param_attr=fluid.ParamAttr(
            name="SparseFeatFactors",
            initializer=fluid.initializer.Uniform()),
   )

sparse_embed_seq = list(map(embedding_layer, inputs[1:-1])) # [C1~C26]
```

#### FC层
将离散数据通过embedding查表得到的值，与连续数据的输入进行`concat`操作，合为一个整体输入，作为全链接层的原始输入。我们共设计了3层FC，每层FC的输出维度都为400，每层FC都后接一个`relu`激活函数，每层FC的初始化方式为符合正态分布的随机初始化，标准差与上一层的输出维度的平方根成反比。
```python
concated = fluid.layers.concat(sparse_embed_seq + inputs[0:1], axis=1)
        
fc1 = fluid.layers.fc(
   input=concated,
   size=400,
   act="relu",
   param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
         scale=1 / math.sqrt(concated.shape[1]))),
)
fc2 = fluid.layers.fc(
   input=fc1,
   size=400,
   act="relu",
   param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
         scale=1 / math.sqrt(fc1.shape[1]))),
)
fc3 = fluid.layers.fc(
   input=fc2,
   size=400,
   act="relu",
   param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
         scale=1 / math.sqrt(fc2.shape[1]))),
)
```
#### Loss及Auc计算
- 预测的结果通过一个输出shape为2的FC层给出，该FC层的激活函数是softmax，会给出每条样本分属于正负样本的概率。
- 每条样本的损失由交叉熵给出，交叉熵的输入维度为[batch_size,2]，数据类型为float，label的输入维度为[batch_size,1]，数据类型为int。
- 该batch的损失`avg_cost`是各条样本的损失之和
- 我们同时还会计算预测的auc，auc的结果由`fluid.layers.auc()`给出，该层的返回值有三个，分别是全局auc: `auc_var`，当前batch的auc: `batch_auc_var`，以及auc_states: `auc_states`，auc_states包含了`batch_stat_pos, batch_stat_neg, stat_pos, stat_neg`信息。`batch_auc`我们取近20个batch的平均，由参数`slide_steps=20`指定，roc曲线的离散化的临界数值设置为4096，由`num_thresholds=2**12`指定。
```
predict = fluid.layers.fc(
            input=fc3,
            size=2,
            act="softmax",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(fc3.shape[1]))),
        )

cost = fluid.layers.cross_entropy(input=predict, label=inputs[-1])
avg_cost = fluid.layers.reduce_sum(cost)
accuracy = fluid.layers.accuracy(input=predict, label=inputs[-1])
auc_var, batch_auc_var, auc_states = fluid.layers.auc(
                                          input=predict,
                                          label=inputs[-1],
                                          num_thresholds=2**12,
                                          slide_steps=20)
```

完成上述组网后，我们最终可以通过训练拿到`avg_cost`与`auc`两个重要指标。
