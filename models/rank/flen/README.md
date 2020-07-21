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

## 简介

[《FLEN: Leveraging Field for Scalable CTR Prediction》](https://arxiv.org/pdf/1911.04690.pdf)文章提出了field-wise bi-interaction pooling技术，解决了在大规模应用特征field信息时存在的时间复杂度和空间复杂度高的困境，同时提出了一种缓解梯度耦合问题的方法dicefactor。该模型已应用于美图的大规模推荐系统中，持续稳定地取得业务效果的全面提升。

本项目在avazu数据集上验证模型效果

## 数据下载及预处理

## 环境

PaddlePaddle 1.7.2

python3.7 

PaddleRec

## 单机训练

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

## 单机预测

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

## 运行

```
python -m paddlerec.run -m paddlerec.models.rank.flen
```

## 模型效果

在样例数据上测试模型

训练：

```
0702 13:38:20.903220  7368 parallel_executor.cc:440] The Program will be executed on CPU using ParallelExecutor, 2 cards are used, so 2 programs are executed in parallel.
I0702 13:38:20.925912  7368 parallel_executor.cc:307] Inplace strategy is enabled, when build_strategy.enable_inplace = True
I0702 13:38:20.933356  7368 parallel_executor.cc:375] Garbage collection strategy is enabled, when FLAGS_eager_delete_tensor_gb = 0
batch: 2, AUC: [0.09090909 0.        ], BATCH_AUC: [0.09090909 0.        ]
batch: 4, AUC: [0.31578947 0.29411765], BATCH_AUC: [0.31578947 0.29411765]
batch: 6, AUC: [0.41333333 0.33333333], BATCH_AUC: [0.41333333 0.33333333]
batch: 8, AUC: [0.4453125  0.44166667], BATCH_AUC: [0.4453125  0.44166667]
batch: 10, AUC: [0.39473684 0.38888889], BATCH_AUC: [0.44117647 0.41176471]
batch: 12, AUC: [0.41860465 0.45535714], BATCH_AUC: [0.5078125  0.54545455]
batch: 14, AUC: [0.43413729 0.42746615], BATCH_AUC: [0.56666667 0.56      ]
batch: 16, AUC: [0.46433566 0.47460087], BATCH_AUC: [0.53       0.59247649]
batch: 18, AUC: [0.44009217 0.44642857], BATCH_AUC: [0.46 0.47]
batch: 20, AUC: [0.42705314 0.43781095], BATCH_AUC: [0.45878136 0.4874552 ]
batch: 22, AUC: [0.45176471 0.46011281], BATCH_AUC: [0.48046875 0.45878136]
batch: 24, AUC: [0.48375    0.48910256], BATCH_AUC: [0.56630824 0.59856631]
epoch 0 done, use time: 0.21532440185546875
PaddleRec Finish
```

预测

```
PaddleRec: Runner single_cpu_infer Begin
Executor Mode: infer
processor_register begin
Running SingleInstance.
Running SingleNetwork.
QueueDataset can not support PY3, change to DataLoader
QueueDataset can not support PY3, change to DataLoader
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from increment_model/0
batch: 20, AUC: [0.49121353], BATCH_AUC: [0.66176471]
batch: 40, AUC: [0.51156463], BATCH_AUC: [0.55197133]
Infer phase2 of 0 done, use time: 0.3941819667816162
PaddleRec Finish
```

