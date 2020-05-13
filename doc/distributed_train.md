# PaddleRec 分布式训练
## 分布式原理基本介绍
> 占位

## 单机代码转分布式代码

> 占位
### 训练代码准备
参数服务器架构，有两个重要的组成部分：Server与Worker。为了启动训练，我们是否要准备两套代码分别运行呢？答案是不需要的。Paddle Fleet API将两者运行的逻辑进行了很好的统一，用户只需使用`fleet.init(role)`就可以判断当前启动的程序扮演server还是worker。使用如下的编程范式，只需10行，便可将单机代码转变为分布式代码：
``` python
role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)

# Define your network, choose your optimizer(SGD/Adam/Adagrad etc.)
strategy = StrategyFactory.create_sync_strategy()
optimizer = fleet.distributed_optimizer(optimizer, strategy)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
if fleet.is_worker():
    fleet.init_worker()
    # run training
    fleet.stop_worker()
```

### 运行环境准备
- Paddle参数服务器模式的训练，目前只支持在`Liunx`环境下运行，推荐使用`ubuntu`或`CentOS`
- Paddle参数服务器模式的前端代码支持`python 2.7`及`python 3.5+`，若使用`Dataset`模式的高性能IO，需使用`python 2.7`
- 使用多台机器进行分布式训练，请确保各自之间可以通过`ip:port`的方式访问`rpc`服务，使用`http/https`代理会导致通信失败
- 各个机器之间的通信耗费应尽量少

假设我们有两台机器，想要在每台机器上分别启动一个`server`进程以及一个`worker`进程，完成2x2（2个参数服务器，2个训练节点）的参数服务器模式分布式训练，按照如下步骤操作。

### 启动server
机器A，IP地址是`10.89.176.11`，通信端口是`36000`，配置如下环境变量后，运行训练的入口程序：
```bash
export PADDLE_PSERVERS_IP_PORT_LIST="10.89.176.11:36000,10.89.176.12:36000"
export TRAINING_ROLE=PSERVER
export POD_IP=10.89.176.11 # node A：10.89.176.11
export PADDLE_PORT=36000
export PADDLE_TRAINERS_NUM=2
python -u train.py --is_cloud=1
```
应能在日志中看到如下输出：

> I0318 21:47:01.298220 188592128 grpc_server.cc:470] Server listening on 127.0.0.1:36000 selected port: 36000

查看系统进程
> 8624 | ttys000 | 0:02.31 | python -u train.py --is_cloud=1

查看系统进程及端口占用：

> python3.7 | 8624 | paddle | 8u | IPv6 | 0xe149b87d093872e5 | 0t0 | TCP |  localhost:36000 (LISTEN)

也可以看到我们的`server`进程8624的确在`36000`端口开始了监听，等待`worker`的通信。

机器B，IP地址是`10.89.176.12`，通信端口是`36000`，配置如下环境变量后，运行训练的入口程序：
```bash
export PADDLE_PSERVERS_IP_PORT_LIST="10.89.176.11:36000,10.89.176.12:36000"
export TRAINING_ROLE=PSERVER
export POD_IP=10.89.176.12 # node B: 10.89.176.12
export PADDLE_PORT=36000
export PADDLE_TRAINERS_NUM=2
python -u train.py --is_cloud=1
```
也可以看到相似的日志输出与进程状况。（进行验证时，请务必确保IP与端口的正确性）

### 启动worker

接下来我们分别在机器A与B上开启训练进程。配置如下环境变量并开启训练进程：

机器A：
```bash
export PADDLE_PSERVERS_IP_PORT_LIST="10.89.176.11:36000,10.89.176.12:36000"
export TRAINING_ROLE=TRAINER
export PADDLE_TRAINERS_NUM=2
export PADDLE_TRAINER_ID=0 # node A：trainer_id = 0
python -u train.py --is_cloud=1
```

机器B：
```bash
export PADDLE_PSERVERS_IP_PORT_LIST="10.89.176.11:36000,10.89.176.12:36000"
export TRAINING_ROLE=TRAINER
export PADDLE_TRAINERS_NUM=2
export PADDLE_TRAINER_ID=1 # node B: trainer_id = 1
python -u train.py --is_cloud=1
```

运行该命令时，若pserver还未就绪，可在日志输出中看到如下信息：
> server not ready, wait 3 sec to retry...
> 
> not ready endpoints:['10.89.176.11:36000', '10.89.176.12:36000']

worker进程将持续等待，直到server开始监听，或等待超时。

当pserver都准备就绪后，可以在日志输出看到如下信息：
> I0317 11:38:48.099179 16719 communicator.cc:271] Communicator start
> 
> I0317 11:38:49.838711 16719 rpc_client.h:107] init rpc client with trainer_id 0

至此，分布式训练启动完毕，将开始训练。


## PaddleRec分布式运行
> 占位
### 本地模拟分布式
> 占位
### MPI集群运行分布式
> 占位
### PaddleCloud集群运行分布式
> 占位
### K8S集群运行分布式
> 占位


