目录
=================

- [目录](#目录)
- [基于PaddleCloud的分布式训练启动方法](#基于paddlecloud的分布式训练启动方法)
  - [使用PaddleRec提交](#使用paddlerec提交)
    - [第一步：运行环境下安装PaddleCloud的Client](#第一步运行环境下安装paddlecloud的client)
    - [第二步：更改模型运行`config.yaml`配置](#第二步更改模型运行configyaml配置)
    - [第三步：增加集群运行`backend.yaml`配置](#第三步增加集群运行backendyaml配置)
      - [MPI集群的Parameter Server模式配置](#mpi集群的parameter-server模式配置)
      - [K8S集群的Collective模式配置](#k8s集群的collective模式配置)
      - [K8S集群的PS-CPU模式配置](#k8s集群的ps-cpu模式配置)
    - [第四步：任务提交](#第四步任务提交)
  - [使用PaddleCloud Client提交](#使用paddlecloud-client提交)
    - [第一步：在`before_hook.sh`里手动安装PaddleRec](#第一步在before_hooksh里手动安装paddlerec)
    - [第二步：在`config.ini`中调整超参](#第二步在configini中调整超参)
    - [第三步：在`job.sh`中上传文件及修改启动命令](#第三步在jobsh中上传文件及修改启动命令)
    - [第四步: 提交任务](#第四步-提交任务)

# 基于PaddleCloud的分布式训练启动方法

> PaddleCloud目前处于百度内部测试推广阶段，将适时推出面向广大用户的公有云版本，欢迎持续关注

## 使用PaddleRec提交

### 第一步：运行环境下安装PaddleCloud的Client

- 环境要求：python > 2.7.5
- 首先在PaddleCloud平台申请`group`的权限，获得计算资源
- 然后在[PaddleCloud client使用手册](http://wiki.baidu.com/pages/viewpage.action?pageId=1017488941#1.%20安装PaddleCloud客户端)下载安装`PaddleCloud-Cli`
- 在PaddleCloud的个人中心获取`AK`及`SK`


### 第二步：更改模型运行`config.yaml`配置

分布式运行首先需要更改`config.yaml`，主要调整以下内容：

- workspace: 调整为在远程节点运行时的工作目录，一般设置为`"./"`即可
- runner_class: 从单机的"train"调整为"cluster_train"，单机训练->分布式训练（例外情况，k8s上单机单卡训练仍然为train，后续支持）
- fleet_mode: 选择参数服务器模式(ps)，或者GPU的all-reduce模式(collective)
- distribute_strategy: 可选项，选择分布式训练的策略，目前只在参数服务器模式下生效，可选项:`sync、asycn、half_async、geo`

配置选项具体参数，可以参考[yaml配置说明](./yaml.md)

以Rank/dnn模型为例

单机训练配置：

```yaml
# workspace
workspace: "paddlerec.models.rank.dnn"

mode: [single_cpu_train]
runner:
- name: single_cpu_train
  class: train
  epochs: 4
  device: cpu
  save_checkpoint_interval: 2 
  save_checkpoint_path: "increment_dnn" 
  init_model_path: "" 
  print_interval: 10
  phases: [phase1]

dataset:
- name: dataloader_train 
  batch_size: 2
  type: DataLoader 
  data_path: "{workspace}/data/sample_data/train"
  sparse_slots: "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
  dense_slots: "dense_var:13"

phase:
- name: phase1
  model: "{workspace}/model.py"
  dataset_name: dataloader_train 
  thread_num: 1
```

分布式的训练配置可以改为：
```yaml
# 改变一：代码上传至节点后，在默认目录下
workspace: "./" 

mode: [ps_cluster]
runner:
- name: ps_cluster
  # 改变二：调整runner的class
  class: cluster_train
  epochs: 4
  device: cpu
  # 改变三 & 四： 指定fleet_mode 与 distribute_strategy
  fleet_mode: ps
  distribute_strategy: async
  save_checkpoint_interval: 2 
  save_checkpoint_path: "increment_dnn" 
  init_model_path: "" 
  print_interval: 10
  phases: [phase1]

dataset:
- name: dataloader_train 
  batch_size: 2
  type: DataLoader 
  # 改变五： 改变数据的读取目录
  # 通常而言，mpi模式下，数据会下载到远程节点执行目录的'./train_data'下， k8s则与挂载位置有关
  data_path: "{workspace}/train_data"
  sparse_slots: "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
  dense_slots: "dense_var:13"

phase:
- name: phase1
  model: "{workspace}/model.py"
  dataset_name: dataloader_train 
  # 分布式训练节点的CPU_NUM环境变量与thread_num相等，多个phase时，取最大的thread_num
  thread_num: 1
```

除此之外，还需关注数据及模型加载的路径，一般而言：
- PaddleCloud MPI集群下，训练数据会下载到节点运行目录的`./train_data/`，测试数据位于`./test_data/`，其他数据及文件可以通过上传到hdfs配置的`thirdparty`后，自动下载到节点运行目录的`./thirdparty/`文件夹下。
- PaddleCloud K8S集群下，hdfs的指定目录会挂载到节点工作目录的`./afs/`

### 第三步：增加集群运行`backend.yaml`配置

分布式训练除了模型的部分调整外，更重要的是加入集群的配置选项，我们通过另一个yaml文件来指定分布式的运行配置，将分布式配置与模型超参解耦。

下面给出一个完整的`backend.yaml`示例：

```yaml
backend: "PaddleCloud"
cluster_type: mpi # k8s 可选

config:
  # 填写任务运行的paddle官方版本号 >= 1.7.2， 默认1.7.2
  paddle_version: "1.7.2" 
  # 是否使用PaddleCloud运行环境下的Python3，默认使用python2
  use_python3: 1

  # hdfs/afs的配置信息填写
  fs_name: "afs://xxx.com"
  fs_ugi: "usr,pwd"

  # 填任务输出目录的远程地址，如afs:/user/your/path/ 则此处填 /user/your/path
  output_path: "" 
  
  # for mpi
  # 填远程数据及地址，如afs:/user/your/path/ 则此处填 /user/your/path
  train_data_path: ""
  test_data_path: "" 
  thirdparty_path: "" 
  
  # for k8s
  # 填远程挂载地址，如afs:/user/your/path/ 则此处填 /user/your/path
  afs_remote_mount_point: "" 

  # paddle参数服务器分布式底层超参，无特殊需求不理不改
  communicator:
    # 使用SGD优化器时，建议设置为1
    FLAGS_communicator_is_sgd_optimizer: 0
    # 以下三个变量默认都等于训练时的线程数：CPU_NUM
    FLAGS_communicator_send_queue_size: 5
    FLAGS_communicator_max_merge_var_num: 5
    FLAGS_communicator_max_send_grad_num_before_recv: 5
    FLAGS_communicator_thread_pool_size: 32
    FLAGS_communicator_fake_rpc: 0
    FLAGS_rpc_retry_times: 3
  
submit:
  # PaddleCloud 个人信息 AK 及 SK
  ak: ""
  sk: ""
  
  # 任务运行优先级，默认high
  priority: "high"
  
  # 任务名称
  job_name: "PaddleRec_CTR"

  # 训练资源所在组
  group: ""

  # 节点上的任务启动命令
  start_cmd: "python -m paddlerec.run -m ./config.yaml"
  
  # 本地需要上传到节点工作目录的文件
  files: ./*.py ./*.yaml

  # for mpi ps-cpu
  # mpi 参数服务器模式下，任务的节点数
  nodes: 2
  
  # for k8s gpu        
  # k8s gpu 模式下，训练节点数，及每个节点上的GPU卡数
  k8s_trainers: 2
  k8s_cpu_cores: 4
  k8s_gpu_card: 1

  # for k8s ps-cpu
  k8s_trainers: 2
  k8s_cpu_cores: 4
  k8s_ps_num: 2
  k8s_ps_cores: 4
  
```

更多backend.yaml配置选项信息，可以查看[yaml配置说明](./yaml.md)

除此之外，我们还需要关注上传到工作目录的文件(`files选项`)的路径问题，在示例中是`./*.py`，说明我们执行任务提交时，与这些py文件在同一目录。若不在同一目录，则需要适当调整files路径，或改为这些文件的绝对路径。

不建议利用`files`上传过大的数据文件，可以通过指定`train_data_path`自动下载，或在k8s模式下指定`afs_remote_mount_point`挂载实现数据到节点的转移。

#### MPI集群的Parameter Server模式配置

下面是一个利用PaddleCloud提交MPI参数服务器模式任务的`backend.yaml`示例

首先调整`config.yaml`:
```yaml
workspace: "./"
mode: [ps_cluster]

dataset:
- name: dataloader_train 
  batch_size: 2
  type: DataLoader 
  data_path: "{workspace}/train_data"
  sparse_slots: "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
  dense_slots: "dense_var:13"

runner:
- name: ps_cluster
  class: cluster_train
  epochs: 2
  device: cpu
  fleet_mode: ps
  save_checkpoint_interval: 1 
  save_checkpoint_path: "increment_dnn" 
  init_model_path: "" 
  print_interval: 1
  phases: [phase1]

phase:
- name: phase1
  model: "{workspace}/model.py"
  dataset_name: dataloader_train 
  thread_num: 1
```


再新增`backend.yaml`
```yaml
backend: "PaddleCloud"
cluster_type: mpi # k8s可选

config:
  paddle_version: "1.7.2" 

  # hdfs/afs的配置信息填写
  fs_name: "afs://xxx.com"
  fs_ugi: "usr,pwd"

  # 填任务输出目录的远程地址，如afs:/user/your/path/ 则此处填 /user/your/path
  output_path: "" 
  
  # for mpi
  # 填远程数据及地址，如afs:/user/your/path/ 则此处填 /user/your/path
  train_data_path: ""
  test_data_path: "" 
  thirdparty_path: "" 

submit:
  # PaddleCloud 个人信息 AK 及 SK
  ak: ""
  sk: ""
  
  # 任务运行优先级，默认high
  priority: "high"
  
  # 任务名称
  job_name: "PaddleRec_CTR"

  # 训练资源所在组
  group: ""

  # 节点上的任务启动命令
  start_cmd: "python -m paddlerec.run -m ./config.yaml"
  
  # 本地需要上传到节点工作目录的文件
  files: ./*.py ./*.yaml

  # for mpi ps-cpu
  # mpi 参数服务器模式下，任务的节点数
  nodes: 2
```

#### K8S集群的Collective模式配置

下面是一个利用PaddleCloud提交K8S集群进行GPU训练的`backend.yaml`示例

首先调整`config.yaml`

```yaml
workspace: "./"
mode: [collective_cluster]

dataset:
- name: dataloader_train 
  batch_size: 2
  type: DataLoader 
  data_path: "{workspace}/afs/挂载数据文件夹的路径"
  sparse_slots: "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
  dense_slots: "dense_var:13"

runner:
- name: collective_cluster
  class: cluster_train
  epochs: 2
  device: gpu
  fleet_mode: collective
  save_checkpoint_interval: 1 # save model interval of epochs
  save_checkpoint_path: "increment_dnn" # save checkpoint path
  init_model_path: "" # load model path
  print_interval: 1
  phases: [phase1]

phase:
- name: phase1
  model: "{workspace}/model.py"
  dataset_name: dataloader_train 
  thread_num: 1
```


再增加`backend.yaml`

```yaml
backend: "PaddleCloud"
cluster_type: k8s # mpi 可选

config:
  # 填写任务运行的paddle官方版本号 >= 1.7.2， 默认1.7.2
  paddle_version: "1.7.2" 

  # hdfs/afs的配置信息填写
  fs_name: "afs://xxx.com"
  fs_ugi: "usr,pwd"

  # 填任务输出目录的远程地址，如afs:/user/your/path/ 则此处填 /user/your/path
  output_path: "" 
  
  # for k8s
  # 填远程挂载地址，如afs:/user/your/path/ 则此处填 /user/your/path
  afs_remote_mount_point: "" 
  
submit:
  # PaddleCloud 个人信息 AK 及 SK
  ak: ""
  sk: ""
  
  # 任务运行优先级，默认high
  priority: "high"
  
  # 任务名称
  job_name: "PaddleRec_CTR"

  # 训练资源所在组
  group: ""

  # 节点上的任务启动命令
  start_cmd: "python -m paddlerec.run -m ./config.yaml"
  
  # 本地需要上传到节点工作目录的文件
  files: ./*.py ./*.yaml
  
  # for k8s gpu        
  # k8s gpu 模式下，训练节点数，及每个节点上的GPU卡数
  k8s_trainers: 2
  k8s_cpu_cores: 4
  k8s_gpu_card: 1
```

#### K8S集群的PS-CPU模式配置
下面是一个利用PaddleCloud提交K8S集群进行参数服务器CPU训练的`backend.yaml`示例

首先调整`config.yaml`:
```yaml
workspace: "./"
mode: [ps_cluster]

dataset:
- name: dataloader_train 
  batch_size: 2
  type: DataLoader 
  data_path: "{workspace}/afs/挂载数据文件夹的路径"
  sparse_slots: "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
  dense_slots: "dense_var:13"

runner:
- name: ps_cluster
  class: cluster_train
  epochs: 2
  device: cpu
  fleet_mode: ps
  save_checkpoint_interval: 1 
  save_checkpoint_path: "increment_dnn" 
  init_model_path: "" 
  print_interval: 1
  phases: [phase1]

phase:
- name: phase1
  model: "{workspace}/model.py"
  dataset_name: dataloader_train 
  thread_num: 1
```

再新增`backend.yaml`
```yaml
backend: "PaddleCloud"
cluster_type: k8s # mpi 可选

config:
  # 填写任务运行的paddle官方版本号 >= 1.7.2， 默认1.7.2
  paddle_version: "1.7.2" 

  # hdfs/afs的配置信息填写
  fs_name: "afs://xxx.com"
  fs_ugi: "usr,pwd"

  # 填任务输出目录的远程地址，如afs:/user/your/path/ 则此处填 /user/your/path
  output_path: "" 
  
  # for k8s
  # 填远程挂载地址，如afs:/user/your/path/ 则此处填 /user/your/path
  afs_remote_mount_point: "" 
  
submit:
  # PaddleCloud 个人信息 AK 及 SK
  ak: ""
  sk: ""
  
  # 任务运行优先级，默认high
  priority: "high"
  
  # 任务名称
  job_name: "PaddleRec_CTR"

  # 训练资源所在组
  group: ""

  # 节点上的任务启动命令
  start_cmd: "python -m paddlerec.run -m ./config.yaml"
  
  # 本地需要上传到节点工作目录的文件
  files: ./*.py ./*.yaml
  
  # for k8s gpu        
  # k8s ps-cpu 模式下，训练节点数，参数服务器节点数，及每个节点上的cpu核心数及内存限制
  k8s_trainers: 2
  k8s_cpu_cores: 4
  k8s_ps_num: 2
  k8s_ps_cores: 4
```

### 第四步：任务提交

当我们准备好`config.yaml`与`backend.yaml`，便可以进行一键任务提交，命令为：

```shell
python -m paddlerec.run -m config.yaml -b backend.yaml
```

执行过程中会进行配置的若干check，并给出错误提示。键入提交命令后，会有以下提交信息打印在屏幕上：

```shell
The task submission folder is generated at /home/PaddleRec/models/rank/dnn/PaddleRec_CTR_202007091308
before_submit
gen gpu before_hook.sh
gen k8s_config.ini
gen k8s_job.sh
gen end_hook.sh
Start checking your job configuration, please be patient.
Congratulations! Job configuration check passed!
Congratulations! The new job is ready for training.
{
    "groupName": "xxxxxxx",
    "jobId": "job-xxxxxx",
    "userId": "x-x-x-x-x"
}
end submit
```

则代表任务已顺利提交PaddleCloud，恭喜。

同时，我们还可以进入`/home/PaddleRec/models/rank/dnn/PaddleRec_CTR_202007091308`这个目录检查我们的提交环境，该目录下有以下文件：

```shell
.
├── backend.yaml         # 用户定义的分布式配置backend.yaml
├── config.yaml          # 用户定义的模型执行config.yaml
├── before_hook.sh       # PaddleRec生成的训练前执行的脚本
├── config.ini           # PaddleRec生成的PaddleCloud环境配置
├── end_hook.sh          # PaddleRec生成的训练后执行的脚本
├── job.sh               # PaddleRec生成的PaddleCloud任务提交脚本
└── model.py             # CTR模型的组网.py文件
```

该目录下的文件会被打平上传到节点的工作目录，用户可以复查PaddleRec生成的配置文件是否符合预期，如不符合预期，既可以调整backend.yaml，亦可以直接修改生成的文件，并执行：

```shell
sh job.sh
```
再次提交任务。


## 使用PaddleCloud Client提交

假如你已经很熟悉PaddleCloud的使用，并且之前是用PaddleCloud-Client提交过任务，熟悉`before_hook.sh`、`config.ini`、`job.sh`，希望通过之前的方式提交PaddleCloud任务，PaddleRec也支持。


我们可以不添加`backend.yaml`，直接用PaddleCloud-Client的提交要求提交任务，除了为分布式训练[修改config.yaml](#第二步更改模型运行configyaml配置)以外，有以下几个额外的步骤：

### 第一步：在`before_hook.sh`里手动安装PaddleRec

```shell
# before_hook.sh
echo "Run before_hook.sh ..."

wget https://paddlerec.bj.bcebos.com/whl/PaddleRec.tar.gz

tar -xf PaddleRec.tar.gz

cd PaddleRec

python setup.py install

echo "End before_hook.sh ..."
```

### 第二步：在`config.ini`中调整超参

```shell
# config.ini
# 设置PADDLE_PADDLEREC_ROLE环境变量为WORKER
# 告诉PaddleRec当前运行环境在节点中，无需执行提交流程，直接执行分布式训练
PADDLE_PADDLEREC_ROLE=WORKER
```

### 第三步：在`job.sh`中上传文件及修改启动命令

我们需要在`job.sh`中上传运行PaddleRec所需的必要文件，如运行该模型的`model.py`、`config.yaml`以及`reader.py`等，PaddleRec的框架代码无需上传，已在before_hook中安装。

同时还需调整启动命令(start_cmd)，调整为
```shell
python -m paddlerec.run -m config.yaml
```

### 第四步: 提交任务

直接运行:

```shell
sh job.sh
```

复用之前的提交脚本执行任务的提交。
