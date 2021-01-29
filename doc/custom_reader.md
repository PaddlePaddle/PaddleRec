# PaddleRec 自定义数据集及Reader

## PaddleRec数据支持方式

### 定长数据

如下两条数据表示定长数据，4个域(label, sparse1, sparse2, dense1)的长度分别是固定的1,2,1,3

```
line1: label:1 sparse1:2 sparse1:3 sparse2:100 dense1:2.1 dense1:5.8 dense1:8.9
line2: label:0 sparse1:78 sparse1:89 sparse2:999 dense1:0.0 dense1:8.8 dense1:7.8
```

对于定长数据(每个特征的表示是固定长度)，动态图模式和静态图模式均支持。

### 变长数据

如下所示，对于sparse1的长度不是固定的，常见于sparse特征域，比如用户标签域，不同的用户的标签数量不同
```
line1: label:1 sparse1:2 sparse1:3 sparse2:100 dense1:2.1 dense1:5.8 dense1:8.9
line2: label:0 sparse1:78 sparse2:999  dense1:0.0 dense1:8.8 dense1:7.8
```

对于变长数据，一种方法是是通过padding的方式补齐成定长，这样动态图静态图均可支持。

由于推荐系统中的变长数据很常见，padding的方式会导致精度和性能。Paddle静态图支持直接读取数据，常见于sparse域，处理方式是embedding后通过sequence_pool转成一个定长的特征。

## 自定义Reader实现

我们提供了两种Reader来读取自定义的数据方式，DataLoader和QueueDataset。

默认是DataLoader模式，可以在runner.reader_type定义两种模式:"DataLoader"或者"QueueDataset"

### DataLoader

我们以下面10条样本组成的简单数据集data/test.txt为例，介绍如何自定义DataLoader，支持动态图和静态图。

```
line1: label:1 sparse1:2 sparse1:3 sparse2:100 dense1:2.1 dense1:5.8 dense1:8.9
line2: label:0 sparse1:78 sparse1:89 sparse2:999 dense1:0.0 dense1:8.8 dense1:7.8
line3: label:1 sparse1:2 sparse1:3 sparse2:100 dense1:2.1 dense1:5.8 dense1:8.9
line4: label:0 sparse1:78 sparse1:89 sparse2:999 dense1:0.0 dense1:8.8 dense1:7.8
...
line10: label:0 sparse1:78 sparse1:89 sparse2:999 dense1:0.0 dense1:8.8 dense1:7.8
```

参照models/rank/dnn 目录下的criteo_reader.py的实现方式

#### 修改xx_reader.py

用户只需要修改class RecDataset中的__iter__函数, 通过python自带的yield方式输出每条数据，目前推荐使用numpy格式输出。

以line1为例 根据自定义函数, 实现对4个特征域的分别输出, yield的格式支持list。
```
yield [numpy.array([1]), numpy.array([2, 3]), numpy.array([100]), numpy.array([2.1,5.8,8.9])]
```
Tips1: 目前的class必须命名为RecDataset, 用户只需要修改__iter__函数

Tips2: 调试过程中可以直接print, 快速调研

#### 修改config.yaml

详细的yaml格式可以参考进阶教程的yaml文档

yaml中的runner.train_reader_path 为训练阶段的reader路径

Tips: importlib格式, 如test_reader.py，则写为train_reader_path: "test_reader"

### QueueDataset

QueueDataset适用于静态图对性能要求特别高的任务，面向大规模数据设计，可以轻松支持亿级的数据读取

#### 修改xx_reader.py

参照models/rank/dnn 目录下的queuedataset_reader.py, 用户需要修改函数generate_sample

Tips: yield返回的dict的序列需要和static_model.py中定义的create_feeds返回的占位符序列保持一致，dict的key值无作用。

#### 修改config.yaml

参照models/rank/config_queuedataset.yaml, 需要将runner.reader_type修改为"QueueDataset", 同时pipe_command修改为"python xx_reader.py"

Tips: pipe_command的执行命令默认是在config.yaml对应的目录下执行
