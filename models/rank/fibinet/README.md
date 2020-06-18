# Fibinet

 以下是本例的简要目录结构及说明： 

```
├── data 
	├── sample_data #样例数据
		├── train
			├── sample_train.txt
	├── download.sh #下载数据
	├── get_slot_data.py #生成slot数据类型
	├── run.sh
├── README.md # 文档
├── model.py #模型文件
├── __init__.py #初始化
├── config.yaml #配置文件
```

## 简介

[《FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction》]( https://arxiv.org/pdf/1905.09433.pdf)是新浪微博机器学习团队发表在RecSys19上的一篇论文，文章指出当前的许多通过特征组合进行CTR预估的工作主要使用特征向量的内积或哈达玛积来计算交叉特征，这种方法忽略了特征本身的重要程度。提出通过使用Squeeze-Excitation network (SENET) 结构动态学习特征的重要性以及使用一个双线性函数来更好的建模交叉特征。
本项目在paddlepaddle上实现FibiNET的网络结构，并在开源数据集Criteo上验证模型效果，

## 数据下载及预处理

数据地址：[Criteo]( https://fleet.bj.bcebos.com/ctr_data.tar.gz)

（1）将原始训练集按9:1划分为训练集和验证集

（2）数值特征（连续特征）进行归一化处理

## 环境

 PaddlePaddle 1.7.2

 python3.7 

 PaddleRec

## 单机训练

CPU环境

在config.yaml文件中设置好设备，epochs等。

```
# select runner by name
mode: [single_cpu_train, single_cpu_infer] #选择模式：cpu训练，cpu预测
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
  save_checkpoint_path: "increment_dnn" # save checkpoint path
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
python -m paddlerec.run -m paddlerec.models.rank.fibinet  
```



## 模型效果

训练：

```
I0618 23:36:48.384914  1052 parallel_executor.cc:440] The Program will be executed on CPU using ParallelExecutor, 1 cards are used, so 1 programs are executed in parallel.
I0618 23:36:48.832674  1052 build_strategy.cc:365] SeqOnlyAllReduceOps:0, num_trainers:1
I0618 23:36:52.915653  1052 parallel_executor.cc:307] Inplace strategy is enabled, when build_strategy.enable_inplace = True
I0618 23:36:53.304608  1052 parallel_executor.cc:375] Garbage collection strategy is enabled, when FLAGS_eager_delete_tensor_gb = 0
batch: 10, AUC: [0.65277778], BATCH_AUC: [0.65277778]
batch: 20, AUC: [0.53877551], BATCH_AUC: [0.46568627]
batch: 30, AUC: [0.32880435], BATCH_AUC: [0.26190476]
epoch 0 done, use time: 9.822723388671875
batch: 10, AUC: [0.42765432], BATCH_AUC: [0.63322884]
batch: 20, AUC: [0.51811594], BATCH_AUC: [0.90931373]
batch: 30, AUC: [0.51916853], BATCH_AUC: [0.91517857]
epoch 1 done, use time: 3.672760009765625
batch: 10, AUC: [0.6169697], BATCH_AUC: [0.97492163]
batch: 20, AUC: [0.66563252], BATCH_AUC: [1.]
batch: 30, AUC: [0.69672379], BATCH_AUC: [1.]
epoch 2 done, use time: 3.920562744140625
batch: 10, AUC: [0.76078133], BATCH_AUC: [1.]
batch: 20, AUC: [0.78663132], BATCH_AUC: [1.]
batch: 30, AUC: [0.81759284], BATCH_AUC: [1.]
epoch 3 done, use time: 3.707246780395508
PaddleRec Finish
```

预测：

```
load persistables from increment_dnn/1
batch: 20, AUC: [0.66411295], BATCH_AUC: [1.]
Infer phase2 of 1 done, use time: 11.975025177001953
load persistables from increment_dnn/3
batch: 20, AUC: [0.87078004], BATCH_AUC: [1.]
Infer phase2 of 3 done, use time: 11.364535570144653
PaddleRec Finish
```

