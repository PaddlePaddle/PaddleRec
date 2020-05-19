# PaddleRec 推荐数据集格式

当你的数据集格式为[slot:feasign]*这种模式，或者可以预处理为这种格式时，可以直接使用PaddleRec内置的Reader。
好处是不用自己写Reader了，各个model之间的数据格式也都可以统一成一样的格式。

## 数据格式说明

假如你的原始数据格式为

```bash
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```

其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。
并且每个特征有一个特征值。
```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔。

假设这13个连续特征（dense slot）的name如下：

```
D1 D2 D3 D4 D4 D6 D7 D8 D9 D10 D11 D12 D13
```

这26个离散特征（sparse slot）的name如下：
```
S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S17 S18 S19 S20 S21 S22 S23 S24 S25 S26
```

那么下面这条样本（1个label + 13个dense值 + 26个feasign）
```
1 0.1 0.4 0.2 0.3 0.5 0.8 0.3 0.2 0.1 0.5 0.6 0.3 0.9 60 16 91 50 52 52 28 69 63 33 87 69 48 59 27 12 95 36 37 41 17 3 86 19 88 60
```

可以转换成：
```
label:1 D1:0.1 D2:0.4 D3:0.2 D4:0.3 D5:0.5 D6:0.8 D7:0.3 D8:0.2 D9:0.1 D10:0.5 D11:0.6 D12:0.3 D13:0.9 S14:60 S15:16 S16:91 S17:50 S18:52 S19:52 S20:28 S21:69 S22:63 S23:33 S24:87 S25:69 S26:48 S27:59 S28:27 S29:12 S30:95 S31:36 S32:37 S33:41 S34:17 S35:3 S36:86 S37:19 S38:88 S39:60
```

注意：上面各个slot:feasign字段之间的顺序没有要求，比如```D1:0.1 D2:0.4```改成```D2:0.4 D1:0.1```也可以。


## 配置

reader中需要配置```sparse_slots```与```dense_slots```，例如

```
  workspace: xxxx

  reader:
    batch_size: 2
    train_data_path: "{workspace}/data/train_data"
    sparse_slots: "label S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 S14 S15 S16 S17 S18 S19 S20 S21 S22 S23 S24 S25 S26"
    dense_slots: "D1:1 D2:1 D3:1 D4:1 D4:1 D6:1 D7:1 D8:1 D9:1 D10:1 D11:1 D12:1 D13:1"

  model:
    xxxxx
```

sparse_slots表示稀疏特征的列表，以空格分开。

dense_slots表示稠密特征的列表，以空格分开。每个字段的格式是```[dense_slot_name]:[dim1,dim2,dim3...]```，其中```dim1,dim2,dim3...```表示shape


配置好了之后，这些slot对应的variable就可以在model中的如下变量啦：
```
self._sparse_data_var

self._dense_data_var
```

# PaddleRec 自定义数据集及Reader

用户自定义数据集及配置异步Reader，需要关注以下几个步骤：

* [数据集整理](#数据集整理)
* [在模型组网中加入输入占位符](#在模型组网中加入输入占位符)
* [Reader实现](#Reader的实现)
* [在yaml文件中配置Reader](#在yaml文件中配置reader)

我们以CTR-DNN模型为例，给出了从数据整理，变量定义，Reader写法，调试的完整历程。

* [数据及Reader示例-DNN](#数据及Reader示例-DNN)


## 数据集整理

PaddleRec支持模型自定义数据集。

关于数据的tips：
1. 数据量：

    PaddleRec面向大规模数据设计，可以轻松支持亿级的数据读取，工业级的数据读写api：`dataset`在搜索、推荐、信息流等业务得到了充分打磨。
2. 文件类型:

    支持任意直接可读的文本数据，`dataset`同时支持`.gz`格式的文本压缩数据，无需额外代码，可直接读取。数据样本应以`\n`为标志，按行组织。

3. 文件存放位置：

    文件通常存放在训练节点本地，但同时，`dataset`支持使用`hadoop`远程读取数据，数据无需下载到本地，为dataset配置hadoop相关账户及地址即可。
4. 数据类型

    Reader处理的是以行为单位的`string`数据，喂入网络的数据需要转为`int`,`float`的数值数据，不支持`string`喂入网络，不建议明文保存及处理训练数据。
5. Tips

    Dataset模式下，训练线程与数据读取线程的关系强相关，为了多线程充分利用，`强烈建议将文件合理的拆为多个小文件`，尤其是在分布式训练场景下，可以均衡各个节点的数据量，同时加快数据的下载速度。

## 在模型组网中加入输入占位符

Reader读取文件后，产出的数据喂入网络，需要有占位符进行接收。占位符在Paddle中使用`fluid.data`或`fluid.layers.data`进行定义。`data`的定义可以参考[fluid.data](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/data_cn.html#data)以及[fluid.layers.data](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn/data_cn.html#data)。

加入您希望输入三个数据，分别是维度32的数据A，维度变长的稀疏数据B，以及一个一维的标签数据C，并希望梯度可以经过该变量向前传递，则示例如下：

数据A的定义：
```python
var_a = fluid.data(name='A', shape= [-1, 32], dtype='float32')
```

数据B的定义，变长数据的使用可以参考[LoDTensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/lod_tensor.html#cn-user-guide-lod-tensor)：
```python
var_b = fluid.data(name='B', shape=[-1, 1], lod_level=1, dtype='int64')
```

数据C的定义：
```python
var_c = fluid.data(name='C', shape=[-1, 1], dtype='int32')
var_c.stop_gradient = False
```

当我们完成以上三个数据的定义后，在PaddleRec的模型定义中，还需将其加入model基类成员变量`self._data_var`

```python
self._data_var.append(var_a)
self._data_var.append(var_b)
self._data_var.append(var_c)
```
至此，我们完成了在组网中定义输入数据的工作。

## Reader的实现

### Reader的实现范式

Reader的逻辑需要一个单独的python文件进行描述。我们试写一个`test_reader.py`，实现的具体流程如下：
1. 首先我们需要引入Reader基类

    ```python
    from paddlerec.core.reader import Reader
    ```
2. 创建一个子类，继承Reader的基类，训练所需Reader命名为`TrainerReader`
    ```python
    class TrainerReader(Reader):
        def init(self):
            pass

        def generator_sample(self, line):
            pass
    ```

3. 在`init(self)`函数中声明一些在数据读取中会用到的变量，必要时可以在`config.yaml`文件中配置变量，利用`env.get_global_env()`拿到。
   
    比如，我们希望从yaml文件中读取一个数据预处理变量`avg=10`，目的是将数据A的数据缩小10倍，可以这样实现：

    首先更改yaml文件，在某个space下加入该变量

    ```yaml
    ...
    train:
        reader:
            avg: 10
    ...
    ```


    再更改Reader的init函数

    ```python
    from paddlerec.core.utils import envs
    class TrainerReader(Reader):
        def init(self):
            self.avg = envs.get_global_env("avg", None, "train.reader")

        def generator_sample(self, line):
            pass
    ```

4. 继承并实现基类中的`generate_sample(self, line)`函数，逐行读取数据。
   - 该函数应返回一个可以迭代的reader方法(带有yield的函数不再是一个普通的函数，而是一个生成器generator，成为了可以迭代的对象，等价于一个数组、链表、文件、字符串etc.)
   - 在这个可以迭代的函数中，如示例代码中的`def reader()`，我们定义数据读取的逻辑。以行为单位的数据进行截取，转换及预处理。
   - 最后，我们需要将数据整理为特定的格式，才能够被PaddleRec的Reader正确读取，并灌入的训练的网络中。简单来说，数据的输出顺序与我们在网络中创建的`inputs`必须是严格一一对应的，并转换为类似字典的形式。
    
    示例： 假设数据ABC在文本数据中，每行以这样的形式存储：
    ```shell
    0.1,0.2,0.3...3.0,3.1,3.2 \t 99999,99998,99997 \t 1 \n
    ```

    则示例代码如下：
    ```python
    from paddlerec.core.utils import envs
    class TrainerReader(Reader):
        def init(self):
            self.avg = envs.get_global_env("avg", None, "train.reader")

        def generator_sample(self, line):
            
            def reader(self, line):
                # 先分割 '\n'， 再以 '\t'为标志分割为list
                variables = (line.strip('\n')).split('\t')

                # A是第一个元素，并且每个数据之间使用','分割
                var_a = variables[0].split(',') # list
                var_a = [float(i) / self.avg for i in var_a] # 将str数据转换为float
                

                # B是第二个元素，同样以 ',' 分割
                var_b = variables[1].split(',') # list
                var_b = [int(i) for i in var_b] # 将str数据转换为int

                # C是第三个元素, 只有一个元素，没有分割符
                var_c = variables[2]
                var_c = int(var_c) # 将str数据转换为int
                var_c = [var_c] # 将单独的数据元素置入list中

                # 将数据与数据名结合，组织为dict的形式
                # 如下，output形式为{ A: var_a, B: var_b, C: var_c}
                variable_name = ['A', 'B', 'C']
                output = zip(variable_name, [var_a] + [var_b] + [var_c])

                # 将数据输出，使用yield方法，将该函数变为了一个可迭代的对象
                yield output

    ```
    
    至此，我们完成了Reader的实现。


### 在yaml文件中配置Reader

在模型的yaml配置文件中，主要的修改是三个，如下

```yaml
reader:
    batch_size: 2
    class: "{workspace}/reader.py"
    train_data_path: "{workspace}/data/train_data"
    reader_debug_mode: False
```

batch_size: 顾名思义，是小批量训练时的样本大小
class: 运行改模型所需reader的路径
train_data_path: 训练数据所在文件夹
reader_debug_mode: 测试reader语法，及输出是否符合预期的debug模式的开关


## 数据及Reader示例-DNN

Reader代码来源于[criteo_reader.py](../models/rank/criteo_reader.py), 组网代码来源于[model.py](../models/rank/dnn/model.py)

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

```


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
