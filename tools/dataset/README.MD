## 数据接入

### 概述

本文档介绍如何在 paddlerec 中数据源的适配和使用方法。

首先需要明确两个问题：
1. 数据是有界的还是无界的，即数据在训练过程中还会同时不断产生还是一次性准备好的固定大小的数据；
2. 数据是通过客户端获取（如存在 kafka、odps、s3 中的数据），还是以文件形式存在于训练代码可读取的目录中；

针对以上问题，本文将介绍 paddle 数据源接入的方法。

### 数据格式

#### 用户数据

典型的用户数据格式为文本形式，本文处理的数据源中的数据将以此为基础，一行数据内容如下所示：

```bash
<label> <float feature 1> ... <float feature 13> <integer feature 1> ... <integer feature 26>
```

其中
* ```<label>```一般用来表示是否被点击，点击用1表示，未点击用0表示
* ```<float feature>```代表连续特征 (dense feature)，示例共有13个连续特征
* ```<integer feature>```代表离散特征 (sparse feature)，示例共有26个离散特征
* 相邻两个特征用```\t```分隔，缺失特征用空格表示

> 如果用户数据与此不一致，请先保持数据格式一致或这在数据处理部分注意对应处理。

#### paddle 数据

paddle 接收的输入数据为 proto (length + 数据) 的形式，具体如
```
3 1234 2345 3456 2 10 21
```
其中 3 为后面 3 个数字组成第一个数据，依次，后面的 2 个数字组成第二个数据。


### 模型组网

了解了需要处理的原始数据格式，对应地在模型组网中，数据将使用如下的形式进行使用

```
dense_input = fluid.layers.data(name="dense_input",
                     shape=[params.dense_feature_dim],
                     dtype="float32")

sparse_input_ids = [
   fluid.layers.data(name="C" + str(i),
                     shape=[1],
                     lod_level=1,
                     dtype="int64") for i in range(1, sparse_feature_num)
   ]

label = fluid.layers.data(name="label", shape=[1], dtype="int64")

model.inputs = [dense_input] + sparse_input_ids + [label]
```

这里的 inputs 定义与数据输入的 proto 形式相对应，顺序很重要，但命名不重要。

### 数据接口

数据需要从数据源中填充至 paddle 模型网络中，使用 QueueDataset 作为数据接口进行训练可以通过如下实现，
```
dataset = paddle.distributed.QueueDataset()
# model.inputs 即为模型网络定义中的数据输入部分，与数据的 proto 形式相对应
dataset.init(use_var=model.inputs, pipe_command="python reader.py", batch_size=batch_size, thread_num=thread_num)
dataset.set_filelist(train_files_list)
exe.train_from_dataset(program,
                       dataset,
                       ...)
```

其中 QueueDataset 主要进行如下处理：
1. 从 set_filelist 指定的 **本地文件** 中读取数据作为初始数据源，如果不使用，请使用真实存在的 fake 文件列表代替；
2. 步骤 1 中的文件会被读取并作为标准输入传递给 pipe_command 中指定的命令进行数据处理;
3. pipe_command 的标准输入将会作为训练数据进入网络进行训练；

需要注意：
* pipe_command 中不能包含与数据无关的输入，如调试日志等；


此外 paddle 还提供 InMemoryDataset 进行数据读取
```
dataset = paddle.distributed.InMemoryDataset()
dataset.load_into_memory()
  train_from_dataset ...
dataset.release_memory()
```

InMemoryDataset 会把数据一次性读取进内存中，然后进行训练，而 QueueDataset 可以实现流式数据加载；


### 数据转换

如上所述，当输入网络模型的输入不是对应形式时，需要对其进行转换。

paddle 提供 fleet.MultiSlotDataGenerator 类抽象中的 generate_sample 方法由用户实现使用。

generate_sample 需要提供数据处理逻辑将数据从原形式转换成如下的 tuple 形式，
```
[("dense_input", [1926, 08, 17]), ("C1", [1111]), ... ("label", [1])]
```
MultiSlotDataGenerator 会负责将该形式的数据转换成对应的 proto 数据形式。

如下示例读取 dataset 中 set_filelist 指定的 filelist，作为标准输入传递给 pipe_command，
即进入 generate_sample 进行处理。

```
class CustomReader(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def wd_reader():
            # line 为从标准输入读取的数据
            features = line.rstrip('\n').split('\t')
            # 数据处理过程，一般包括归一化处理逻辑
            # 从一条文本数据解析并拼装成如下形式
            input_data = [dense_feature]+sparse_feature+[label]
            
            feature_name = ["dense_input"]
            for idx in categorical_range_:
                feature_name.append("C" + str(idx - 13))
            feature_name.append("label")

            yield zip(feature_name, input_data)

        return wd_reader

data_generator = CustomReader()
# 从标准输入读取数据
data_generator.run_from_stdin()

```

如果不使用标准输入，可以使用用户自定义的 genenrator 产生产生数据，
示例如下，

```
class CustomReader(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def wd_reader():
            for line in customGenerator:
                # 数据处理逻辑一致
                ...
                yield result

        return wd_reader

data_generator = CustomReader()
data_generator.run_from_memory()

```

这里的 customGenerator 需要实现一个 python 的 generator，具体如下
```python
class CustomGenerator:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        # 返回一条数据 
        return line
```
使用自定义的 generator 可以使用客户端从不同的数据源如 kafka、s3 中读取数据。

### 总结

总结 paddle 中使用数据接入的步骤如下：
#### 本地文件

这里的本地文件指所有能够通过 mount 挂载形成在程序中直接通过目录进行读取的方式，接入流程如下

1. 文本文件无需特殊处理，特殊格式如 zip、npz 等数据需要准备解析函数，文件目录通过 set_filelist 设置；
2. 定义 CustomReader，实现 generate_sample 函数对数据按照前述进行解析；函数输出 tuple 类型的解析后的数据，注意顺序需要与组网顺序一致；

#### 数据源

数据源指需要通过客户端进行数据读取的方式，接入流程如下

1. 准备获取数据源的客户端，如 kafka 客户端，s3 客户端，能够读取数据 print 单行数据；将客户端包装成 CustomGenerator;
2. 定义 CustomReader，实现 generate_sample 函数对数据按照前述进行解析；函数输出 tuple 类型的解析后的数据，注意顺序需要与组网顺序一致；
