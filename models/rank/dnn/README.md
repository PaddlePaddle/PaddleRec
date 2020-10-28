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
sh run.sh
```
进入models/rank/dnn/data目录下，执行该脚本，会从国内源的服务器上下载Criteo数据集，并解压到指定文件夹。原始的全量数据放置于`./train_data_full/`，原始的全量测试数据放置于`./test_data_full/`，原始的用于快速验证的训练数据与测试数据放置于`./train_data/`与`./test_data/`。处理后的全量训练数据放置于`./slot_train_data_full/`，处理后的全量测试数据放置于`./slot_test_data_full/`，处理后的用于快速验证的训练数据与测试数据放置于`./slot_train_data/`与`./slot_test_data/`。

执行该脚本的理想输出为：
```bash
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
6. 最后，我们需要将数据整理为特定的格式，才能够被dataset正确读取，并灌入的训练的网络中。简单来说，数据的输出顺序与我们在网络中创建的`inputs`必须是严格一一对应的。在示例代码中，我们将数据整理成`click:value dense_feature:value ... dense_feature:value 1:value ... 26:value`的格式。用print输出是因为我们在run.sh中将结果重定向到slot_train_data等文件中，由模型直接读取。在用户自定义使用时，可以使用`zip`的方法将参数名与数值构成的元组组成了一个list，并将其yield输出，并在config.yaml中的data_converter参数指定reader的路径。


```python
import paddle.fluid.incubate.data_generator as dg

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


class CriteoDataset(dg.MultiSlotDataGenerator):
    """
    DacDataset: inheritance MultiSlotDataGeneratior, Implement data reading
    Help document: http://wiki.baidu.com/pages/viewpage.action?pageId=728820675
    """

    def generate_sample(self, line):
        """
        Read the data line by line and process it as a dictionary
        """

        def reader():
            """
            This function needs to be implemented by the user, based on data format
            """
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
            s = "click:" + str(label[0])
            for i in dense_feature:
                s += " dense_feature:" + str(i)
            for i in range(1, 1 + len(categorical_range_)):
                s += " " + str(i) + ":" + str(sparse_feature[i - 1][0])
            print(s.strip()) # add print for data preprocessing

        return reader


d = CriteoDataset()
d.run_from_stdin()
```
### 快速调试Dataset
我们可以脱离组网架构，单独验证Dataset的输出是否符合我们预期。使用命令
`cat 数据文件 | python dataset读取python文件`进行dataset代码的调试：
```bash
cat train_data/part-0 | python get_slot_data.py
```
输出的数据格式如下：
`label:value dense_input:value ... dense_input:value sparse_input:value ... sparse_input:value `

理想的输出为(截取了一个片段)：
```bash
...
click:0 dense_feature:0.05 dense_feature:0.00663349917081 dense_feature:0.05 dense_feature:0.0 dense_feature:0.02159375 dense_feature:0.008 dense_feature:0.15 dense_feature:0.04 dense_feature:0.362 dense_feature:0.1 dense_feature:0.2 dense_feature:0.0 dense_feature:0.04 1:715353 2:817085 3:851010 4:833725 5:286835 6:948614 7:881652 8:507110 9:27346 10:646986 11:643076 12:200960 13:18464 14:202774 15:532679 16:729573 17:342789 18:562805 19:880474 20:984402 21:666449 22:26235 23:700326 24:452909 25:884722 26:787527
...
```

## 模型组网
### 数据输入声明
正如数据准备章节所介绍，Criteo数据集中，分为连续数据与离散（稀疏）数据，所以整体而言，CTR-DNN模型的数据输入层包括三个，分别是：`dense_input`用于输入连续数据，维度由超参数`dense_input_dim`指定，数据类型是归一化后的浮点型数据。`sparse_inputs`用于记录离散数据，在Criteo数据集中，共有26个slot，所以我们创建了名为`1~26`的26个稀疏参数输入，数据类型为整数；最后是每条样本的`label`，代表了是否被点击，数据类型是整数，0代表负样例，1代表正样例。

### CTR-DNN模型组网

CTR-DNN模型的组网比较直观，本质是一个二分类任务，代码参考`model.py`。模型主要组成是一个`Embedding`层，四个`FC`层，以及相应的分类任务的loss计算和auc计算。

#### Embedding层
首先介绍Embedding层的搭建方式：`Embedding`层的输入是`sparse_input`，由超参的`sparse_feature_number`和`sparse_feature_dimshape`定义。需要特别解释的是`is_sparse`参数，当我们指定`is_sprase=True`后，计算图会将该参数视为稀疏参数，反向更新以及分布式通信时，都以稀疏的方式进行，会极大的提升运行效率，同时保证效果一致。

各个稀疏的输入通过Embedding层后，将其合并起来，置于一个list内，以方便进行concat的操作。

```python
def embedding_layer(input):
    if self.distributed_embedding:
        emb = fluid.contrib.layers.sparse_embedding(
            input=input,
            size=[self.sparse_feature_number, self.sparse_feature_dim],
            param_attr=fluid.ParamAttr(
                name="SparseFeatFactors",
                initializer=fluid.initializer.Uniform()))
    else:
        emb = fluid.layers.embedding(
            input=input,
            is_sparse=True,
            is_distributed=self.is_distributed,
            size=[self.sparse_feature_number, self.sparse_feature_dim],
            param_attr=fluid.ParamAttr(
                name="SparseFeatFactors",
                initializer=fluid.initializer.Uniform()))
    emb_sum = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    return emb_sum

sparse_embed_seq = list(map(embedding_layer, self.sparse_inputs)) # [C1~C26]
```

#### FC层
将离散数据通过embedding查表得到的值，与连续数据的输入进行`concat`操作，合为一个整体输入，作为全链接层的原始输入。我们共设计了4层FC，每层FC的输出维度由超参`fc_sizes`指定，每层FC都后接一个`relu`激活函数，每层FC的初始化方式为符合正态分布的随机初始化，标准差与上一层的输出维度的平方根成反比。
```python
concated = fluid.layers.concat(
    sparse_embed_seq + [self.dense_input], axis=1)

fcs = [concated]
hidden_layers = envs.get_global_env("hyper_parameters.fc_sizes")

for size in hidden_layers:
    output = fluid.layers.fc(
        input=fcs[-1],
        size=size,
        act='relu',
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Normal(
                scale=1.0 / math.sqrt(fcs[-1].shape[1]))))
    fcs.append(output)

```
#### Loss及Auc计算
- 预测的结果通过一个输出shape为2的FC层给出，该FC层的激活函数是softmax，会给出每条样本分属于正负样本的概率。
- 每条样本的损失由交叉熵给出，交叉熵的输入维度为[batch_size,2]，数据类型为float，label的输入维度为[batch_size,1]，数据类型为int。
- 该batch的损失`avg_cost`是各条样本的损失之和
- 我们同时还会计算预测的auc，auc的结果由`fluid.layers.auc()`给出，该层的返回值有三个，分别是从第一个batch累计到当前batch的全局auc: `auc`，最近几个batch的auc: `batch_auc`，以及auc_states: `_`，auc_states包含了`batch_stat_pos, batch_stat_neg, stat_pos, stat_neg`信息。`batch_auc`我们取近20个batch的平均，由参数`slide_steps=20`指定，roc曲线的离散化的临界数值设置为4096，由`num_thresholds=2**12`指定。
```
predict = fluid.layers.fc(
    input=fcs[-1],
    size=2,
    act="softmax",
    param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
        scale=1 / math.sqrt(fcs[-1].shape[1]))))

self.predict = predict

auc, batch_auc, _ = fluid.layers.auc(input=self.predict,label=self.label_input,
                                     num_thresholds=2**12,
                                     slide_steps=20)

cost = fluid.layers.cross_entropy(
            input=self.predict, label=self.label_input)
avg_cost = fluid.layers.reduce_mean(cost)
```

完成上述组网后，我们最终可以通过训练拿到`BATCH_AUC`与`auc`两个重要指标。

### 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。 在全量数据下模型的指标如下：
| 模型 | auc | batch_size | thread_num| epoch_num| Time of each epoch |
| :------| :------ | :------| :------ | :------| :------ | 
| dnn | 0.7748 | 512 | 10 | 4 | 约3.5小时 |

1. 确认您当前所在目录为PaddleRec/models/rank/dnn  
2. 在data目录下运行数据一键处理脚本，处理时间较长，请耐心等待。命令如下：  
``` 
cd data
sh run.sh
cd ..
```
3. 退回dnn目录中，打开文件config.yaml,更改其中的参数  
将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  
将dataloader_train中的batch_size从2改为512  
将dataloader_train中的data_path改为{workspace}/data/slot_train_data_full  
将dataset_infer中的batch_size从2改为512  
将dataset_infer中的data_path改为{workspace}/data/slot_test_data_full  
根据自己的需求调整phase中的线程数  
4. 运行命令，模型会进行四个epoch的训练，然后预测第四个epoch，并获得相应auc指标  
```
python -m paddlerec.run -m ./config.yaml
```
5. 经过全量数据训练后，执行预测的结果示例如下：

```
PaddleRec: Runner single_cpu_infer Begin
Executor Mode: infer
processor_register begin
Running SingleInstance.
Running SingleNetwork.
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from increment_dnn/3
batch: 20, BATCH_AUC: [0.75670043], AUC: [0.77490453]
batch: 40, BATCH_AUC: [0.77020144], AUC: [0.77490437]
batch: 60, BATCH_AUC: [0.77464683], AUC: [0.77490435]
batch: 80, BATCH_AUC: [0.76858989], AUC: [0.77490416]
batch: 100, BATCH_AUC: [0.75728286], AUC: [0.77490362]
batch: 120, BATCH_AUC: [0.75007016], AUC: [0.77490286]
...
batch: 720, BATCH_AUC: [0.76840144], AUC: [0.77489881]
batch: 740, BATCH_AUC: [0.76659033], AUC: [0.77489854]
batch: 760, BATCH_AUC: [0.77332639], AUC: [0.77489849]
batch: 780, BATCH_AUC: [0.78361653], AUC: [0.77489874]
Infer phase2 of epoch increment_dnn/3 done, use time: 52.7707588673, global metrics: BATCH_AUC=[0.78361653], AUC=[0.77489874]
PaddleRec Finish
```

## 流式训练（OnlineLearning）任务启动及配置流程

### 流式训练简介
流式训练是按照一定顺序进行数据的接收和处理，每接收一个数据，模型会对它进行预测并对当前模型进行更新，然后处理下一个数据。 像信息流、小视频、电商等场景，每天都会新增大量的数据， 让每天(每一刻)新增的数据基于上一天(上一刻)的模型进行新的预测和模型更新。

在大规模流式训练场景下， 需要使用的深度学习框架有对应的能力支持， 即：
* 支持大规模分布式训练的能力， 数据量巨大， 需要有良好的分布式训练及扩展能力，才能满足训练的时效要求
* 支持超大规模的Embedding， 能够支持十亿甚至千亿级别的Embedding, 拥有合理的参数输出的能力，能够快速输出模型参数并和线上其他系统进行对接
* Embedding的特征ID需要支持HASH映射，不要求ID的编码，能够自动增长及控制特征的准入(原先不存在的特征可以以适当的条件创建)， 能够定期淘汰(能够以一定的策略进行过期的特征的清理) 并拥有准入及淘汰策略
* 最后就是要基于框架开发一套完备的流式训练的 trainer.py， 能够拥有完善的流式训练流程

### 使用ctr-dnn online learning 进行模型的训练
目前，PaddleRec基于飞桨分布式训练框架的能力，实现了这套流式训练的流程。 供大家参考和使用。我们基于`models/rank/ctr-dnn`修改了一个online_training的版本，供大家更好的理解和参考。

**注意**
1. 使用online learning 需要安装目前Paddle最新的开发者版本， 你可以从 https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-dev 此处获得它，需要先卸载当前已经安装的飞桨版本，根据自己的Python环境下载相应的安装包。
2. 使用online learning 需要安装目前PaddleRec最新的开发者版本， 你可以通过 git clone https://github.com/PaddlePaddle/PaddleRec.git 得到最新版的PaddleRec并自行安装

### 启动方法
1. 修改config.yaml中的 hyper_parameters.distributed_embedding=1，表示打开大规模稀疏的模式
2. 修改config.yaml中的 mode: [single_cpu_train, single_cpu_infer] 中的 `single_cpu_train` 为online_learning_cluster，表示使用online learning对应的运行模式
3. 准备训练数据， ctr-dnn中使用的online learning对应的训练模式为 天级别训练， 每天又分为24个小时， 因此训练数据需要 天--小时的目录结构进行整理。
    以 2020年08月10日 到 2020年08月11日 2天的训练数据举例， 用户需要准备的数据的目录结构如下：
    ```
    train_data/
    |-- 20200810
    |   |-- 00
    |   |   `-- train.txt
    |   |-- 01
    |   |   `-- train.txt
    |   |-- 02
    |   |   `-- train.txt
    |   |-- 03
    |   |   `-- train.txt
    |   |-- 04
    |   |   `-- train.txt
    |   |-- 05
    |   |   `-- train.txt
    |   |-- 06
    |   |   `-- train.txt
    |   |-- 07
    |   |   `-- train.txt
    |   |-- 08
    |   |   `-- train.txt
    |   |-- 09
    |   |   `-- train.txt
    |   |-- 10
    |   |   `-- train.txt
    |   |-- 11
    |   |   `-- train.txt
    |   |-- 12
    |   |   `-- train.txt
    |   |-- 13
    |   |   `-- train.txt
    |   |-- 14
    |   |   `-- train.txt
    |   |-- 15
    |   |   `-- train.txt
    |   |-- 16
    |   |   `-- train.txt
    |   |-- 17
    |   |   `-- train.txt
    |   |-- 18
    |   |   `-- train.txt
    |   |-- 19
    |   |   `-- train.txt
    |   |-- 20
    |   |   `-- train.txt
    |   |-- 21
    |   |   `-- train.txt
    |   |-- 22
    |   |   `-- train.txt
    |   `-- 23
    |       `-- train.txt
    `-- 20200811
        |-- 00
        |   `-- train.txt
        |-- 01
        |   `-- train.txt
        |-- 02
        |   `-- train.txt
        |-- 03
        |   `-- train.txt
        |-- 04
        |   `-- train.txt
        |-- 05
        |   `-- train.txt
        |-- 06
        |   `-- train.txt
        |-- 07
        |   `-- train.txt
        |-- 08
        |   `-- train.txt
        |-- 09
        |   `-- train.txt
        |-- 10
        |   `-- train.txt
        |-- 11
        |   `-- train.txt
        |-- 12
        |   `-- train.txt
        |-- 13
        |   `-- train.txt
        |-- 14
        |   `-- train.txt
        |-- 15
        |   `-- train.txt
        |-- 16
        |   `-- train.txt
        |-- 17
        |   `-- train.txt
        |-- 18
        |   `-- train.txt
        |-- 19
        |   `-- train.txt
        |-- 20
        |   `-- train.txt
        |-- 21
        |   `-- train.txt
        |-- 22
        |   `-- train.txt
        `-- 23
            `-- train.txt
    ```    
4. 准备好数据后， 即可按照标准的训练流程进行流式训练了
    ```shell
    python -m paddlerec.run -m models/rank/dnn/config.yaml
    ```
