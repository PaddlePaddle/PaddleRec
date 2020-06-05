# 模型调参

PaddleRec模型调参需要同时关注两个部分
1. model.py
2. config.yaml 中 hyper_parameters的部分

我们以`models/rank/dnn`为例介绍两者的对应情况：

```yaml
hyper_parameters:
  optimizer:
    class: Adam
    learning_rate: 0.001
  sparse_feature_number: 1000001
  sparse_feature_dim: 9
  fc_sizes: [512, 256, 128, 32]
```

## optimizer

该参数决定了网络参数训练时使用的优化器，class可选项有：`SGD`/`Adam`/`AdaGrad`，通过learning_rate选项设置学习率。

在`PaddleRec/core/model.py`中，可以看到该选项是如何生效的：

```python
if name == "SGD":
    reg = envs.get_global_env("hyper_parameters.reg", 0.0001,
                                self._namespace)
    optimizer_i = fluid.optimizer.SGD(
        lr, regularization=fluid.regularizer.L2DecayRegularizer(reg))
elif name == "ADAM":
    optimizer_i = fluid.optimizer.Adam(lr, lazy_mode=True)
elif name == "ADAGRAD":
    optimizer_i = fluid.optimizer.Adagrad(lr)
```


## sparse_feature_number & sparse_feature_dim

该参数指定了ctr-dnn组网中，Embedding表的维度，在`PaddelRec/models/rank/dnn/model.py`中可以看到该参数是如何生效的：

```python
self.sparse_feature_number = envs.get_global_env(
            "hyper_parameters.sparse_feature_number")
self.sparse_feature_dim = envs.get_global_env(
            "hyper_parameters.sparse_feature_dim")

def embedding_layer(input):
    emb = fluid.layers.embedding(
        input=input,
        is_sparse=True,
        is_distributed=self.is_distributed,
        size=[self.sparse_feature_number, self.sparse_feature_dim],
        param_attr=fluid.ParamAttr(
            name="SparseFeatFactors",
            initializer=fluid.initializer.Uniform()), )
    emb_sum = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    return emb_sum
```

## fc_sizes

该参数指定了ctr-dnn模型中的dnn共有几层，且每层的维度，在在`PaddelRec/models/rank/dnn/model.py`中可以看到该参数是如何生效的：

```python
hidden_layers = envs.get_global_env("hyper_parameters.fc_sizes")

for size in hidden_layers:
    output = fluid.layers.fc(
        input=fcs[-1],
        size=size,
        act='relu',
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Normal(
                scale=1.0 / math.sqrt(fcs[-1].shape[1]))))
    fcs.append(output)
```
