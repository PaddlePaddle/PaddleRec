# PaddleRec 预训练模型

PaddleRec基于业务实践，使用真实数据，产出了推荐领域算法的若干预训练模型，方便开发者进行算法调研。

## 文本分类预训练模型

### 获取地址

```bash
wget https://paddlerec.bj.bcebos.com/textcnn_pretrain%2Fpretrain_model.tar.gz
```

### 使用方法

解压后，得到的是一个paddle的模型文件夹，使用`PaddleRec/models/contentunderstanding/textcnn`模型进行加载  
您可以在PaddleRec/models/contentunderstanding/textcnn_pretrain中找到finetune_startup.py文件，在config.yaml中配置startup_class_path和init_pretraining_model_path两个参数。  
在参数startup_class_path中配置finetune_startup.py文件的地址，在init_pretraining_model_path参数中配置您要加载的参数文件。  
以textcnn_pretrain为例，配置完的runner如下：
```
runner:
- name: train_runner
  class: train
  epochs: 6
  device: cpu
  save_checkpoint_interval: 1
  save_checkpoint_path: "increment"
  init_model_path: "" 
  print_interval: 10
  startup_class_path: "{workspace}/finetune_startup.py"
  init_pretraining_model_path: "{workspace}/pretrain_model/pretrain_model_params"
  phases: phase_train
```
具体使用方法请参照textcnn[使用预训练模型进行finetune](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/contentunderstanding/textcnn_pretrain)
