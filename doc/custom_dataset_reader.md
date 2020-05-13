# PaddleRec 自定义数据集及Reader

## 数据集及reader配置简介

以`ctr-dnn`模型举例：

```yaml
reader:
    batch_size: 2
    class: "{workspace}/../criteo_reader.py"
    train_data_path: "{workspace}/data/train"
    reader_debug_mode: False
```
有以上4个需要重点关注的配置选项：

- batch_size: 网络进行小批量训练的一组数据的大小
- class: 指定数据处理及读取的`reader` python文件
- train_data_path: 训练数据所在地址
- reader_debug_mode: 测试reader语法，及输出是否符合预期的debug模式的开关

## 自定义数据集

PaddleRec支持模型自定义数据集，在model.config.yaml文件中的reader部分，通过`train_data_path`指定数据读取路径。

关于数据的tips

- PaddleRec 面向的是推荐与搜索领域，数据以文本格式为主
- Dataset模式支持读取文本数据压缩后的`.gz`格式
- Dataset模式下，训练线程与数据读取线程的关系强相关，为了多线程充分利用，`强烈建议将文件拆成多个小文件`，尤其是在分布式训练场景下，可以均衡各个节点的数据量。

## 自定义Reader

数据集准备就绪后，需要适当修改或重写一个新的reader以适配数据集或新组网。

我们以`ctr-dnn`网络举例`reader`的正确打开方式，网络文件位于`models/rank/dnn`。

### Criteo数据集格式

CTR-DNN训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```bash
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。

### Criteo数据集的预处理

数据预处理共包括两步：
- 将原始训练集按9:1划分为训练集和验证集
- 数值特征（连续特征）需进行归一化处理，但需要注意的是，对每一个特征```<integer feature i>```，归一化时用到的最大值并不是用全局最大值，而是取排序后95%位置处的特征值作为最大值，同时保留极值。

### CTR网络输入的定义

正如前所述，Criteo数据集中，分为连续数据与离散（稀疏）数据，所以整体而言，CTR-DNN模型的数据输入层包括三个，分别是：`dense_input`用于输入连续数据，维度由超参数`dense_feature_dim`指定，数据类型是归一化后的浮点型数据。`sparse_input_ids`用于记录离散数据，在Criteo数据集中，共有26个slot，所以我们创建了名为`C1~C26`的26个稀疏参数输入，并设置`lod_level=1`，代表其为变长数据，数据类型为整数；最后是每条样本的`label`，代表了是否被点击，数据类型是整数，0代表负样例，1代表正样例。

在Paddle中数据输入的声明使用`paddle.fluid.layers.data()`，会创建指定类型的占位符，数据IO会依据此定义进行数据的输入。

稀疏参数输入的定义:
```python
def sparse_inputs():
    ids = envs.get_global_env("hyper_parameters.sparse_inputs_slots", None, self._namespace)

    sparse_input_ids = [
        fluid.layers.data(name="S" + str(i),
                            shape=[1],
                            lod_level=1,
                            dtype="int64") for i in range(1, ids)
    ]
    return sparse_input_ids
```

稠密参数输入的定义：
```python
def dense_input():
    dim = envs.get_global_env("hyper_parameters.dense_input_dim", None, self._namespace)

    dense_input_var = fluid.layers.data(name="D",
                                        shape=[dim],
                                        dtype="float32")
    return dense_input_var
```

标签的定义：
```python
def label_input():
    label = fluid.layers.data(name="click", shape=[1], dtype="int64")
    return label
```

组合起来，正确的声明他们：
```python
self.sparse_inputs = sparse_inputs()
self.dense_input = dense_input()
self.label_input = label_input()

self._data_var.append(self.dense_input)

for input in self.sparse_inputs:
    self._data_var.append(input)

self._data_var.append(self.label_input)

if self._platform != "LINUX":
    self._data_loader = fluid.io.DataLoader.from_generator(
        feed_list=self._data_var, capacity=64, use_double_buffer=False, iterable=False)
```
若运行于**Linux**环境下，默认使用**dataset**模式读取数据集；若运行于**windows**或**mac**下，默认使用**dataloader**模式读取数据集。以上两种方法是paddle.io中提供的不同模式，`dataset`运行速度更快，但依赖于linux的环境，因此会有该逻辑判断。

> Paddle的组网中不支持数据输入为`str`类型，`强烈不建议使用明文保存和读取数据`

### Criteo Reader写法

```python
# 引入PaddleRec的Reader基类
from paddlerec.core.reader import Reader
# 引入PaddleRec的读取yaml配置文件的方法
from paddlerec.core.utils import envs

# 定义TrainReader，需要继承 paddlerec.core.reader.Reader
class TrainReader(Reader):

    # 数据预处理逻辑，继承自基类
    # 如果无需处理， 使用pass跳过该函数的执行
    def init(self):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
        self.hash_dim_ = envs.get_global_env("hyper_parameters.sparse_feature_number", None, "train.model")
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)

    # 读取数据方法，继承自基类
    # 实现可以迭代的reader函数，逐行处理数据
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
            for idx in self.continuous_range_:
                if features[idx] == "":
                    dense_feature.append(0.0)
                else:
                    dense_feature.append(
                        (float(features[idx]) - self.cont_min_[idx - 1]) /
                        self.cont_diff_[idx - 1])

            for idx in self.categorical_range_:
                sparse_feature.append(
                    [hash(str(idx) + features[idx]) % self.hash_dim_])
            label = [int(features[0])]
            feature_name = ["D"]
            for idx in self.categorical_range_:
                feature_name.append("S" + str(idx - 13))
            feature_name.append("label")
            yield zip(feature_name, [dense_feature] + sparse_feature + [label])

        return reader
```

### 如何自定义数据读取规则

在上文我们看到了由`criteo_reader.py`实现具体的数据读取规则，那么，怎样为自己的数据集写规则呢？

具体流程如下：
1. 首先我们需要引入Reader基类

    ```python
    from paddlerec.core.reader import Reader
    ```
2. 创建一个子类，继承Reader的基类，训练所需Reader命名为`TrainerReader`
3. 在`init(self)`函数中声明一些在数据读取中会用到的变量，如示例代码中的`cont_min_`、`categorical_range_`等，必要时可以在`config.yaml`文件中配置变量，通过`env.get_global_env()`拿到。
4. 继承并实现基类中的`generate_sample(self, line)`函数，逐行读取数据。该函数应返回一个可以迭代的reader方法(带有yield的函数不再是一个普通的函数，而是一个生成器generator，成为了可以迭代的对象，等价于一个数组、链表、文件、字符串etc.)
5. 在这个可以迭代的函数中，如示例代码中的`def reader()`，我们定义数据读取的逻辑。以行为单位的数据进行截取，转换及预处理。
6. 最后，我们需要将数据整理为特定的格式，才能够被dataset正确读取，并灌入的训练的网络中。简单来说，数据的输出顺序与我们在网络中创建的`inputs`必须是严格一一对应的，并转换为类似字典的形式。在示例代码中，我们使用`zip`的方法将参数名与数值构成的元组组成了一个list，并将其yield输出。如果展开来看，我们输出的数据形如

    `[('dense_feature',[value]),('C1',[value]),('C2',[value]),...,('C26',[value]),('label',[value])]`


### 调试Reader

在Linux下运行时，默认启动`Dataset`模式，在Win/Mac下运行时，默认启动`Dataloader`模式。

通过在`config.yaml`中添加或修改`reader_debug_mode=True`打开debug模式，只会结合组网运行reader的部分，读取10条样本，并print，方便您观察格式是否符合预期或隐藏bug。
```yaml
reader:
    batch_size: 2
    class: "{workspace}/../criteo_reader.py"
    train_data_path: "{workspace}/data/train"
    reader_debug_mode: True
```

修改后，使用paddlerec.run执行该修改后的yaml文件，可以观察输出。
```bash
python -m paddlerec.run -m ./models/rank/dnn/config.yaml -e single
```

### Dataset调试

dataset输出的数据格式如下：
` dense_input:size ; dense_input:value ; sparse_input:size ; sparse_input:value ; ... ; sparse_input:size ; sparse_input:value ; label:size ; label:value `

基本规律是对于每个变量，会先输出其维度大小，再输出其具体值。

直接debug `criteo_reader`理想的输出为(截取了一个片段)：
```bash
...
13 0.0 0.00497512437811 0.05 0.08 0.207421875 0.028 0.35 0.08 0.082 0.0 0.4 0.0 0.08 1 737395 1 210498 1 903564 1 286224 1 286835 1 906818 1 90
6116 1 67180 1 27346 1 51086 1 142177 1 95024 1 157883 1 873363 1 600281 1 812592 1 228085 1 35900 1 880474 1 984402 1 100885 1 26235 1 410878 1 798162 1 499868 1 306163 1 0
...
```
可以看到首先输出的是13维的dense参数，随后是分立的sparse参数，最后一个是1维的label，数值为0，输出符合预期。

>使用Dataset的一些注意事项
> - Dataset的基本原理：将数据print到缓存，再由C++端的代码实现读取，因此，我们不能在dataset的读取代码中，加入与数据读取无关的print信息，会导致C++端拿到错误的数据信息。
> - dataset目前只支持在`unbuntu`及`CentOS`等标准Linux环境下使用，在`Windows`及`Mac`下使用时，会产生预料之外的错误，请知悉。

### DataLoader调试

dataloader的输出格式为`list: [ list[var_1], list[var_2], ... , list[var_3]]`，每条样本的数据会被放在一个 **list[list]** 中，list[0]为第一个variable。

直接debug `criteo_reader`理想的输出为(截取了一个片段)：
```bash
...
[[0.0, 0.004975124378109453, 0.05, 0.08, 0.207421875, 0.028, 0.35, 0.08, 0.082, 0.0, 0.4, 0.0, 0.08], [560746], [902436], [262029], [182633], [368411], [735166], [321120], [39572], [185732], [140298], [926671], [81559], [461249], [728372], [915018], [907965], [818961], [850958], [311492], [980340], [254960], [175041], [524857], [764893], [526288], [220126], [0]]
...
```
可以看到首先输出的是13维的dense参数的list，随后是分立的sparse参数，各自在一个list中，最后一个是1维的label的list，数值为0，输出符合预期。
