# FLEN

 以下是本例的简要目录结构及说明： 

```
├── data #样例数据
	├── sample_data
		├── train
			├── sample_train.txt
	├── run.sh
	├── get_slot_data.py
├── __init__.py
├── README.md # 文档
├── model.py #模型文件
├── config.yaml #配置文件
```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)


---
## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [论文复现](#论文复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)

## 模型简介

[《FLEN: Leveraging Field for Scalable CTR Prediction》](https://arxiv.org/pdf/1911.04690.pdf)文章提出了field-wise bi-interaction pooling技术，解决了在大规模应用特征field信息时存在的时间复杂度和空间复杂度高的困境，同时提出了一种缓解梯度耦合问题的方法dicefactor。该模型已应用于美图的大规模推荐系统中，持续稳定地取得业务效果的全面提升。

本项目在avazu数据集上验证模型效果, 本模型配置默认使用demo数据集，若进行精度验证，请参考[论文复现](#论文复现)部分。

本项目支持功能

训练：单机CPU、单机单卡GPU、单机多卡GPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)   

预测：单机CPU、单机单卡GPU ；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md) 

## 数据准备




## 运行环境

PaddlePaddle>=1.7.2 

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos



## 快速开始
### 单机训练

CPU环境

在config.yaml文件中设置好设备，epochs等。

```
# select runner by name
mode: [single_cpu_train, single_cpu_infer]
# config of each runner.
# runner is a kind of paddle training class, which wraps the train/infer process.
runner:
- name: single_cpu_train
  class: train
  # num of epochs
  epochs: 4
  # device to run training or infer
  device: cpu
  save_checkpoint_interval: 2 # save model interval of epochs
  save_inference_interval: 4 # save inference
  save_checkpoint_path: "increment_model" # save checkpoint path
  save_inference_path: "inference" # save inference path
  save_inference_feed_varnames: [] # feed vars of save inference
  save_inference_fetch_varnames: [] # fetch vars of save inference
  init_model_path: "" # load model path
  print_interval: 10
  phases: [phase1]
```

### 单机预测

CPU环境

在config.yaml文件中设置好epochs、device等参数。

```
- name: single_cpu_infer
  class: infer
  # num of epochs
  epochs: 1
  # device to run training or infer
  device: cpu #选择预测的设备
  init_model_path: "increment_dnn" # load model path
  phases: [phase2]
```

### 运行

```
python -m paddlerec.run -m paddlerec.models.rank.flen
```

## 论文复现

用原论文的完整数据复现论文效果需要在config.yaml中修改batch_size=512, thread_num=8, epoch_num=1

全量数据的效果未来补充。




## 进阶使用

## FAQ
