# 数据接入

本文介绍 paddlerec 中数据源的适配和接入方法。

本文提供的接入方法适用于以下数据形式
* 以文本形式存在的固定大小文件数据，如 data.txt, data.csv;
* 以特殊格式存储、需要程序解析的文件，如 data.npz, data.gz, data.zip;
* 需要使用客户端从远端获取的数据文件，如 http://example.com/data.txt;
* 需要使用专用协议/客户端获取的数据或数据流，如存储于 kafka, s3 中的数据；

上述数据形式在进行数据处理前都会被转换为单行的文本数据形式，其他数据形式也可以参考本文中的处理方法进行适配。

### 概述

先通过下面这个简单但完整的例子来认识 paddle 数据处理的流程，对整个流程有整体认识

```python
import paddle
import os

paddle.enable_static()

with open("test_queue_dataset_run_a.txt", "w") as f:
    data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
    data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
    data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
    data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
    f.write(data)
with open("test_queue_dataset_run_b.txt", "w") as f:
    data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
    data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
    data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
    data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
    f.write(data)

slots = ["slot1", "slot2", "slot3", "slot4"]
slots_vars = []
for slot in slots:
    var = paddle.static.data(
        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
    slots_vars.append(var)

dataset = paddle.distributed.InMemoryDataset()
dataset.init(
    batch_size=1,
    thread_num=2,
    input_type=1,
    pipe_command="cat",
    use_var=slots_vars)
dataset.set_filelist(
    ["test_queue_dataset_run_a.txt", "test_queue_dataset_run_b.txt"])
dataset.load_into_memory()

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
startup_program = paddle.static.Program()
main_program = paddle.static.default_main_program()
exe.run(startup_program)

exe.train_from_dataset(main_program, dataset)

os.remove("./test_queue_dataset_run_a.txt")
os.remove("./test_queue_dataset_run_b.txt")
```

保留关键流程部分

```python
slots_vars = [...]                                # paddle 数据输入层，可以理解为 data 或 tensor 的占位符，详细见 模型组网 部分
dataset = paddle.distributed.InMemoryDataset()    # 选用 dataset， 详细见 数据接口 部分
dataset.set_filelist(["..."])                     # 设置输入文件，详细见 数据接口 部分
dataset.init(...
    pipe_command="cat",                           # 数据处理 shell 程序，详细见 数据转换 部分
    use_var=slots_vars)                           # dataset 读取的数据会被填充至 slots_vars
exe.train_from_dataset(main_program, dataset)     
```

以下将对上述示例中的各部分进行详细介绍和拓展。

### 数据格式

#### paddle 数据

上面的例子中使用了 paddle 能够直接读取的数据格式，即数据在进入 paddle 网络输入层的时候需要转换成特定的 proto 格式，具体为

```bash
[ids_num id1 id2 ...] ...
```
这是一种先声明数据个数，后面跟数据的形式，如
```
3 1234 2345 3456 2 10 21
```
表示 3 为后面 3 个数字组成第一个数据，依次，后面的 2 个数字组成第二个数据，该数据对应的输入层应为一个 1x3 的 tensor 和一个 1x2 的 tensor。

更加复杂的数据格式支持请参考 LoDTensor 相关文档。

#### 用户数据

典型的用户数据格式为文本形式，本文处理的数据源中的数据将以此为基础，一行数据内容如下所示：

```bash
<label> <dense feature 1> ... <dense feature 13> <sparse feature 1> ... <sparse feature 26>
```
例如
```bash
0 0 0 10 5 1673 91 15 33 256 0 5  5 75ac2fe6 04e09220 b1ecc6c4 5dff9b29 25c83c98 7e0ccccf 63282fe3 0b153874 a73ee510 b95c890d e6959f26 2436ff75 b57fa159 07d13a8f f6b23a53 f4ead43c 8efede7f 6fc84bfb   4f1aa25f ad3062eb 423fab69 ded4aac9
```

其中
* ```<label>```一般用来表示是否被点击，点击用1表示，未点击用0表示
* ```<dense feature>```代表连续特征，示例共有13个连续特征
* ```<sparse feature>```代表离散特征，示例共有26个离散特征
* 相邻两个特征用 ```\t``` 分隔，缺失特征用空格表示

> 上述数据格式不固定，可以根据用户自己的数据对后续数据解析部分进行对应处理即可，不必要转换为上述形式。

至此，明确了一般用户数据的格式和 paddle 程序需要的数据格式，将用户数据转换为 paddle 数据即为本文主要内容，基本路径为通过使用 paddle 提供的 dataset 将数据输出为 proto 格式。


### 模型组网

在上述例子中仅仅展示了一组 tensor 作为占位符，真实场景中会提供一个网络，网络的外层输入即为例子中 use_var 需要填充的值。
在 wide and deep 模型中，输入层被分为三组，如下所示

```python
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
其中，
* dense_input 为 1 个 13 维的 tensor，占据一个 slot，对应的填充 proto 形式为  13 x x x x x x x x x x x x x 这样的形式；
* sparse_input_ids 为 26 个 1 维的 tensor，占据 26 个 slot，对应的填充 proto 序列为 1 x 1 x 1 x 1 x ... 这样的形式；
* label 为 1 个 1 维 tensor，占据 1 个 slot，对应填充 proto 为 1 1 这样的形式；

> 这里的 inputs 定义与数据输入的 proto 形式相对应，顺序很重要，与 dataset 输入必须对应；但命名 name 不重要，但要注意这里的 name 可以作为存储模型在 inference 中的对应关系使用。

### 数据接口

数据需要从数据源中填充至 paddle 模型网络中，使用 QueueDataset 作为数据接口进行训练可以通过如下实现，
```python
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

特别注意：
> pipe_command 中不能包含与数据无关的输入，如调试日志等；

此外 paddle 还提供 InMemoryDataset 进行数据读取
```python
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
```shell
[("dense_input", [1926, 08, 17]), ("C1", [1111]), ... ("label", [1])]
```
MultiSlotDataGenerator 会负责将该形式的数据转换成对应的 proto 数据形式。

如下示例读取 dataset 中 set_filelist 指定的 filelist，作为标准输入传递给 pipe_command，
即进入 generate_sample 进行处理。

```python
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

```python
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

> 定义 CustomReader，实现 generate_sample 函数对数据按照前述进行解析；函数输出 tuple 类型的解析后的数据，注意顺序需要与组网顺序一致；

### 总结

1. paddle 使用的填充数据形式为 proto 形式，通过 dataset 进行训练时，要保证其输出形式正确且顺序对应；
2. dataset 提供 InMemoryDataset 和 QueueDataset 可以使用，区别在于后者可以处理无界数据；
3. dataset 提供文件读取功能可以直接读取文本，并作为标准输入给 pipe_command 指定的 shell 程序；
4. 用户通过扩充 pipe_command 以实现复杂的数据处理流程，用户可选用任意语言实现数据处理逻辑；
5. 使用 python 处理时，paddle 提供了 fleet.MultiSlotDataGenerator 做为基类供用户使用，用户只需实现 generator 即可；
6. 特殊的数据源也可以通过将客户端包装成 CustomGenerator 来实现数据输入；

### demo 和 测试

本文附件中的数据处理程序可将用户数据转换成 proto 形式，可使用 shell 进行直接测试：
```shell
# test file reader

python file_reader.py < utils/part-0

# test cpp reader 

g++ -o parser parser.cpp && ./parser < utils/part-0

# test tfrecord reader

python tfrecord_reader.py < utils/wd.tfrecord

# test kafka reader

export KAFKA_HOSTS=1.1.1.1
export KAFKA_GID=xxx
export KAFKA_TOPICS=wide-and-deep-data

python kafka_reader.py

# test odps reader 
#
# create config.py 
#

python odps_reader.py

```

