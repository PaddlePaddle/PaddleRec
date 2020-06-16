# 如何添加自定义模型

当您希望开发自定义模型时，需要继承模型的模板基类，并实现三个必要的方法`init_hyper_parameter`,`intput_data`,`net`

并按照以下规范添加代码。

### 基类的继承

继承`paddlerec.core.model`的ModelBase，命名为`Class Model`

```python
from paddlerec.core.model import ModelBase


class Model(ModelBase):

    # 构造函数无需显式指定
    # 若继承，务必调用基类的__init__方法
    def __init__(self, config):
        ModelBase.__init__(self, config)
        # ModelBase的__init__方法会调用_init_hyper_parameter()
    
```

### 超参的初始化

继承并实现`_init_hyper_parameter`方法(必要)，可以在该方法中，从`yaml`文件获取超参或进行自定义操作。如下面的示例：

所有的envs调用接口在_init_hyper_parameters()方法中实现，同时类成员也推荐在此做声明及初始化。

```python
 def _init_hyper_parameters(self):
    self.feature_size = envs.get_global_env(
        "hyper_parameters.feature_size")
    self.expert_num = envs.get_global_env("hyper_parameters.expert_num")
    self.gate_num = envs.get_global_env("hyper_parameters.gate_num")
    self.expert_size = envs.get_global_env("hyper_parameters.expert_size")
    self.tower_size = envs.get_global_env("hyper_parameters.tower_size")
```


### 数据输入的定义
继承并实现`input_data`方法(非必要)


#### 直接使用基类的数据读取方法

`ModelBase`中的input_data默认实现为slot_reader，在`config.yaml`中分别配置`reader.sparse_slot`及`reader.dense_slot`选项实现`slog:feasign`模式的数据读取。

> Slot : Feasign 是什么？
>
> Slot直译是槽位，在Rec工程中，是指某一个宽泛的特征类别，比如用户ID、性别、年龄就是Slot，Feasign则是具体值，比如：12345，男，20岁。
> 
> 在实践过程中，很多特征槽位不是单一属性，或无法量化并且离散稀疏的，比如某用户兴趣爱好有三个：游戏/足球/数码，且每个具体兴趣又有多个特征维度，则在兴趣爱好这个Slot兴趣槽位中，就会有多个Feasign值。
>
> PaddleRec在读取数据时，每个Slot ID对应的特征，支持稀疏，且支持变长，可以非常灵活的支持各种场景的推荐模型训练。

使用示例请参考`rank.dnn`模型。

#### 自定义数据输入


如果您不想使用`slot:feasign`模式，则需继承并实现`input_data`接口，接口定义：`def input_data(self, is_infer=False, **kwargs)`

使用示例如下：

```python
def input_data(self, is_infer=False, **kwargs):
    ser_slot_names = fluid.data(
        name='user_slot_names',
        shape=[None, 1],
        dtype='int64',
        lod_level=1)
    item_slot_names = fluid.data(
        name='item_slot_names',
        shape=[None, self.item_len],
        dtype='int64',
        lod_level=1)
    lens = fluid.data(name='lens', shape=[None], dtype='int64')
    labels = fluid.data(
        name='labels',
        shape=[None, self.item_len],
        dtype='int64',
        lod_level=1)
 
    train_inputs = [user_slot_names] + [item_slot_names] + [lens] + [labels]
    infer_inputs = [user_slot_names] + [item_slot_names] + [lens]
 
    if is_infer:
        return infer_inputs
    else:
        return train_inputs
```

更多数据读取教程，请参考[自定义数据集及Reader](custom_dataset_reader.md)


### 组网的定义

继承并实现`net`方法(必要)

- 接口定义`def net(self, inputs, is_infer=False)`
- 自定义网络需在该函数中使用paddle组网，实现前向逻辑，定义网络的Loss及Metrics，通过`is_infer`判断是否为infer网络。
- 我们强烈建议`train`及`infer`尽量复用相同代码，
- `net`中调用的其他函数以下划线为头进行命名，封装网络中的结构模块，如`_sparse_embedding_layer(self)`。
- `inputs`为`def input_data()`的输出，若使用`slot_reader`方式，inputs为占位符，无实际意义，通过以下方法拿到dense及sparse的输入

  ```python
  self.sparse_inputs = self._sparse_data_var[1:]
  self.dense_input = self._dense_data_var[0]
  self.label_input = self._sparse_data_var[0]
  ```

可以参考官方模型的示例学习net的构造方法。

## 如何运行自定义模型

记录`model.py`,`config.yaml`及数据读取`reader.py`的文件路径，建议置于同一文件夹下，如`/home/custom_model`下，更改`config.yaml`中的配置选项

1. 更改 workerspace为模型文件所在文件夹 
```yaml
workspace: "/home/custom_model"
```

2. 更改数据地址及读取reader地址
```yaml
dataset:
- name: custom_model_train
- data_path: "{workspace}/data/train" # or  "/home/custom_model/data/train"
- data_converter: "{workspace}/reader.py" # or "/home/custom_model/reader.py"
```

3. 更改执行器的路径配置
```yaml
mode: train_runner

runner:
- name: train_runner
  class: train
  device: cpu
  epochs: 10
  save_checkpoint_interval: 2
  save_inference_interval: 5
  save_checkpoint_path: "{workspace}/increment" # or  "/home/custom_model/increment"
  save_inference_path: "{workspace}/inference" # or  "/home/custom_model/inference"
  print_interval: 10

phase:
- name: train
  model: "{workspace}/model.py" # or "/home/custom_model/model"
  dataset_name: custom_model_train
  thread_num: 1
```

4. 使用paddlerec.run方法运行自定义模型

```shell
python -m paddlerec.run -m /home/custom_model/config.yaml 
```

以上~请开始享受你的推荐算法高效开发流程。如有任何问题，欢迎在[issue](https://github.com/PaddlePaddle/PaddleRec/issues)提出，我们会第一时间跟进解决。
