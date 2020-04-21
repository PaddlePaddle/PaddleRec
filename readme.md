# 快速开始

## 环境准备
Fleet-Rec是基于飞桨分布式训练所开发的，包含模型、训练模式的快速开发、调试、部署的工具， 让用户更轻松的使用飞桨分布式训练。

- 安装飞桨  **注：需要用户安装最新版本的飞桨<当前只支持Linux系统>。**

```bash
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

- 安装Fleet-Rec

```
git clone https://github.com/seiriosPlus/FleetRec/
cd FleetRec
python setup.py install
```

## ctr-dnn示例使用
目前框架内置了ctr-dnn模型，后续会加入更多模型

示例代码位于FleetRec/fleetrec/example/下， 当前支持单机训练和本地1*1模拟训练

### 单机训练
```bash
cd FleetRec

python -m fleetrec.run \
       -m fleetrec/examples/ctr-dnn_train.yaml \
       -e single 
```

### 本地模拟分布式训练

```bash
cd FleetRec

python -m fleetrec.run \
       -m fleetrec/examples/ctr-dnn_train.yaml \
       -e local_cluster 
```

### 集群提交分布式训练<需要用户预先配置好集群环境，本提交命令不包含提交客户端>

```bash
cd FleetRec

python -m fleetrec.run \
       -m fleetrec/examples/ctr-dnn_train.yaml \
       -e cluster
```

更多用户文档及二次开发文档，敬请期待。