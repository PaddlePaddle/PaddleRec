# PaddleRec 流式训练（OnlineLearning）任务启动及配置流程


## 流式训练简介
流式训练是按照一定顺序进行数据的接收和处理，每接收一个数据，模型会对它进行预测并对当前模型进行更新，然后处理下一个数据。 像信息流、小视频、电商等场景，每天都会新增大量的数据， 让每天(每一刻)新增的数据基于上一天(上一刻)的模型进行新的预测和模型更新。

在大规模流式训练场景下， 需要使用的深度学习框架有对应的能力支持， 即：
* 支持大规模分布式训练的能力， 数据量巨大， 需要有良好的分布式训练及扩展能力，才能满足训练的时效要求
* 支持超大规模的Embedding， 能够支持十亿甚至千亿级别的Embedding, 拥有合理的参数输出的能力，能够快速输出模型参数并和线上其他系统进行对接
* Embedding的特征ID需要支持HASH映射，不要求ID的编码，能够自动增长及控制特征的准入(原先不存在的特征可以以适当的条件创建)， 能够定期淘汰(能够以一定的策略进行过期的特征的清理) 并拥有准入及淘汰策略
* 最后就是要基于框架开发一套完备的流式训练的 trainer.py， 能够拥有完善的流式训练流程

## 使用PaddleRec内置的 online learning 进行模型的训练
目前，PaddleRec基于飞桨分布式训练框架的能力，实现了这套流式训练的流程。 供大家参考和使用。我们在`models/online_learning`目录下提供了一个ctr-dnn的online_training的版本，供大家更好的理解和参考。

**注意**
1. 使用online learning 需要安装目前Paddle最新的开发者版本， 你可以从 https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-dev 此处获得它，需要先卸载当前已经安装的飞桨版本，根据自己的Python环境下载相应的安装包。
2. 使用流式训练及大规模稀疏的能力，需要对模型做一些微调， 因此需要你修改部分代码。
3. 当前只有参数服务器的分布式训练是支持带大规模稀疏的流式训练的，因此运行时，请直接选择参数服务器本地训练或集群训练方法。

## 启动方法

### 1. 启动内置模型的默认配置训练

在安装好`paddlepaddle`及`paddlerec`后，可以直接使用一行命令快速启动内置模型的默认配置训练，命令如下;

```shell
python -m paddlerec.run -m paddlerec.models.xxx.yyy
```

注意事项：
1. 请确保调用的是安装了paddlerec的`python`环境
2. `xxx`为paddlerec.models下有多个大类，如：`recall`/`rank`/`rerank`等
3. `yyy`为每个类别下又有多个模型，如`recall`下有：`gnn`/`grup4rec`/`ncf`等

例如启动`recall`下的`word2vec`模型的默认配置;

```shell
python -m paddlerec.run -m models/recall/word2vec
```

### 2. 启动内置模型的个性化配置训练

如果我们修改了默认模型的config.yaml文件，怎么运行修改后的模型呢？

- **没有改动模型组网**

  假如你将paddlerec代码库克隆在了`/home/PaddleRec`，并修改了`/home/PaddleRec/models/rank/dnn/config.yaml`，则如下启动训练

  ```shell
  python -m paddlerec.run -m /home/PaddleRec/models/rank/dnn/config.yaml
  ```

  paddlerec 运行的是在paddlerec库安装目录下的组网文件(model.py)，但个性化配置`config.yaml`是用的是指定路径下的yaml文件。

- **改动了模型组网**

  假如你将paddlerec代码库克隆在了`/home/PaddleRec`，并修改了`/home/PaddleRec/models/rank/dnn/model.py`， 以及`/home/PaddleRec/models/rank/dnn/config.yaml`，则首先需要更改`yaml`中的`workspace`的设置：

  ```yaml
  workspace: /home/PaddleRec/models/rank/dnn/
  ```

  再执行：

  ```shell
  python -m paddlerec.run -m /home/PaddleRec/models/rank/dnn/config.yaml
  ```

  paddlerec 运行的是绝对路径下的组网文件(model.py)以及个性化配置文件(config.yaml)




## yaml训练配置

### yaml中训练相关的概念

`config.yaml`中训练流程相关有两个重要的逻辑概念，`runner`与`phase`：

- **`runner`** : runner是训练的引擎，亦可称之为运行器，在runner中定义执行设备（cpu、gpu），执行的模式（训练、预测、单机、多机等），以及运行的超参，例如训练轮数，模型保存地址等。
- **`phase`** : phase是训练中的阶段的概念，是引擎具体执行的内容，该内容是指：具体运行哪个模型文件，使用哪个reader。

PaddleRec每次运行时，会执行一个或多个运行器，通过`mode`指定`runner`的名字。每个运行器可以执行一个或多个`phase`，所以PaddleRec支持一键启动多阶段的训练。

### 单机CPU训练

下面我们开始定义一个单机CPU训练的`runner`:

```yaml
mode: single_cpu_train # 执行名为 single_cpu_train 的运行器
# mode 也支持多个runner的执行，此处可以改为 mode: [single_cpu_train, single_cpu_infer]

runner:
- name: single_cpu_train # 定义 runner 名为 single_cpu_train
  class: train # 执行单机训练
  device: cpu # 执行在 cpu 上
  epochs: 10 # 训练轮数

  save_checkpoint_interval: 2 # 每隔2轮保存一次checkpoint
  save_inference_interval: 4 # 每隔4轮保存一次inference model
  save_checkpoint_path: "increment" # checkpoint 的保存地址
  save_inference_path: "inference" # inference model 的保存地址
  save_inference_feed_varnames: [] # inference model 的feed参数的名字
  save_inference_fetch_varnames: [] # inference model 的fetch参数的名字
  init_model_path: "" # 如果是加载模型热启，则可以指定初始化模型的地址
  print_interval: 10 # 训练信息的打印间隔，以batch为单位
  phases: [phase_train] # 若没有指定phases，则会默认运行所有phase
  # phase 也支持自定多个phase的执行，此处可以改为 phases: [phase_train, phase_infer]
```

再定义具体的执行内容：

```yaml
phase:
- name: phase_train # 该阶段名为 phase1 
  model: "{workspace}/model.py" # 模型文件为workspace下的model.py
  dataset_name: dataset_train # reader的名字

dataset:
- name: dataset_train
  type: DataLoader # 使用DataLoader的数据读取方式
  batch_size: 2
  data_path: "{workspace}/train_data" # 数据地址
  sparse_slots: "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26" # sparse 输入的位置定义
  dense_slots: "dense_var:13"  # dense参数的维度定义

```

### 单机单卡GPU训练

具体执行内容与reader与前述相同，下面介绍需要改动的地方

```yaml
mode: single_gpu_train # 执行名为 single_gpu_train 的运行器

runner:
- name: single_gpu_train # 定义 runner 名为 single_gpu_train
  class: train # 执行单机训练
  device: gpu # 执行在 gpu 上
  selected_gpus: "0" # 默认选择在id=0的卡上执行训练
  epochs: 10 # 训练轮数
```

### 单机多卡GPU训练

具体执行内容与reader与前述相同，下面介绍需要改动的地方

```yaml
mode: single_multi_gpu_train # 执行名为 single_multi_gpu_train 的运行器

runner:
- name: single_multi_gpu_train # 定义 runner 名为 single_multi_gpu_train
  class: train # 执行单机训练
  device: gpu # 执行在 gpu 上
  selected_gpus: "0,1,2,3" # 选择多卡执行训练
  epochs: 10 # 训练轮数
```

### 本地模拟参数服务器训练
具体执行内容与reader与前述相同，下面介绍需要改动的地方

```yaml
mode: local_cluster_cpu_train # 执行名为 local_cluster_cpu_train 的运行器

runner:
- name: local_cluster_cpu_train # 定义 runner 名为 runner_train
  class: local_cluster_train # 执行本地模拟分布式——参数服务器训练
  device: cpu # 执行在 cpu 上（paddle后续版本会支持PS-GPU）
  worker_num: 1 # (可选)worker进程数量，默认1
  server_num: 1 # (可选)server进程数量，默认1
  epochs: 10 # 训练轮数
```
