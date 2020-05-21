# PaddleRec 分布式训练

paddlerec支持依托`k8s`进行分布式训练，同时支持`PaddleCloud`及`mpi`集群等环境的大规模分布式任务提交。

## 分布式任务提交原理

paddlerec在运行时，通过额外的`-b`启动命令指定分布式运行的配置`backend.yaml`。以k8s为提交示例：
```yaml
engine:
  backend: "k8s"
  job_name: "k8s-ctr-dnn"

  submit:
    server_num: 2
    trainer_num: 2
    docker_image: "hub.baidubce.com/ctr/paddlerec:alpha"
    memory: "4Gi"
    storage: "10Gi"
    log_level: 0
```

首先最重要的配置是`backend`选项，指定了分布式运行的集群，例如：`k8s`,`PaddleCloud`,`MPI`等，paddlerec会根据该选项调用不同的分布式提交脚本，如k8s的启动脚本[cluster.sh](../core/engine/cluster/k8s/cluster.sh)，进行不同的分布式流程。

`job_name`是分布式任务提交的唯一区分标识，有关的日志及配置文件，以该名称进行构造及保存。

`submit`下的各项配置是各分布式集群所特有的选项，其`backend_yaml`有所区别，可以通过不同示例了解提交流程。

分布式训练及预测教程将随版本迭代不断更新，欢迎关注。如有任何问题或建议，欢迎在[Github Issue](https://github.com/PaddlePaddle/PaddleRec/issues)提出。

### K8S集群运行分布式

我们以在百度云K8S集群提交分布式任务为例，介绍云上分布式的运行方法，主要分为以下几个步骤

1. 申请百度云K8S集群资源
2. 申请存储资源
3. 本地配置`kubectl`
4. k8s集群配置`volcano`任务调度系统
5. 本地配置集群信息
6. 修改任务`config.yaml`
7. 增加分布式配置文件`backend.yaml`

1、2步骤不是paddlerec的重点，我们暂且跳过，从第3步开始说明。

#### 本地配置kubectl

[kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/#before-you-begin)是Kubernetes集群的命令行工具，使用kubectl非常方便的管理k8s集群，进行任务的提交。

我们需要首先在本地开发机环境中部署`kubectl`:[Linux安装kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/#before-you-begin)


#### 配置volcano

[volcano](https://github.com/volcano-sh/volcano)是基于k8s的批处理任务调度系统，它提供了许多批处理和弹性工作负载类通常需要的一套机制，非常适合于分布式深度学习的开发。

可以使用`kubectl`进行快速配置安装：
```bash
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml
```

#### 本地配置集群信息

当拥有k8s集群的管理权限后，将生成的`config`文件复制到kubectl的默认配置文件夹：

```bash
cp config ./kube/config
```

#### 修改任务yaml

为进行分布式训练，我们需要修改模型`config.yaml`的配置

```yaml
train:
  epochs: 10
  # engine: single 
  engine: cluster
  workspace: "paddlerec.models.rank.dnn"

  trainer:
    # for cluster training
    strategy: "async"
    threads: 2
```
关键配置的含义：

- engine: 执行器的选择，分布式训练填写cluster
- stratege: 参数服务器分布式运行模式的选择，有sync/async/half_async/geo等选择
- threads: 分布式训练时各个节点的线程数

#### 增加分布式配置

我们同时还需对分布式运行的一些超参进行配置，新增一个`backend.yaml`文件对分布式超参进行描述，我们以`rank.dnn`模型为例进行介绍：

`backend.yaml`文件配置如下：

```yaml
engine:
  backend: "k8s"
  job_name: "k8s-ctr-dnn"

  submit:
    server_num: 2
    trainer_num: 2
    docker_image: "hub.baidubce.com/ctr/paddlerec:alpha"
    memory: "4Gi"
    storage: "10Gi"
    log_level: 0
```

各个配置的含义为：

- backend: 分布式集群的区分flag，使用k8s集群填写k8s即可
- job_name: 本次任务的标识名
- server_num: 参数服务器模式中参数服务器节点的数量
- worker_num: 参数服务器模式中训练节点的数量
- docker_image: k8s使用的docker镜像的来源及版本(建议使用百度bce)
- mermory: 每个节点的内存上限
- storage: 每个节点的存储空间上限
- log_level: paddle运行中GLOG的等级，建议为0，需要进行开发者模型调试OP时可适当调整

paddlerec会根据以上配置，使用模板[k8s.yaml.template](../core/engine/cluster/k8s/k8s.yaml.template)生成k8s任务所需的`k8s.yaml`，进而依赖`kubectl`完成任务的提交。

#### 执行k8s分布式训练

当`config.yaml`与`backend.yaml`修改完毕后，执行以下命令提交分布式训练任务：

```bash
python -m paddlerec.run -m config.yaml -b backend.yaml
```