# fibinet

 以下是本例的简要目录结构及说明： 

```
├── data #样例数据
	├── sample_data
		├── train
			├── sample_train.txt
	├── download.sh
	├── run.sh
	├── get_slot_data.py
├── __init__.py
├── README.md # 文档
├── model.py #模型文件
├── config.yaml #配置文件
```

## 简介

[《FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction》]( https://arxiv.org/pdf/1905.09433.pdf)是新浪微博机器学习团队发表在RecSys19上的一篇论文，文章指出当前的许多通过特征组合进行CTR预估的工作主要使用特征向量的内积或哈达玛积来计算交叉特征，这种方法忽略了特征本身的重要程度。提出通过使用Squeeze-Excitation network (SENET) 结构动态学习特征的重要性以及使用一个双线性函数来更好的建模交叉特征。

本项目在paddlepaddle上实现FibiNET的网络结构，并在开源数据集Criteo上验证模型效果。

## 数据下载及预处理

数据地址：[Criteo]( https://fleet.bj.bcebos.com/ctr_data.tar.gz)

（1）将原始训练集按9:1划分为训练集和验证集

（2）数值特征（连续特征）进行归一化处理

执行run.sh生成训练集和测试集

```
sh run.sh
```

原始的数据格式为13个dense部分特征+离散化特征，用'\t'切分
```
0   1   1   5   0   1382    4   15  2   181 1   2       2   68fd1e64    80e26c9b    fb936136    7b4723c4    25c83c98    7e0ccccf    de7995b8    1f89b562    a73ee510    a8cd5504    b2cb9c98    37c9c164    2824a5f6    1adce6ef    8ba8b39a    891b62e7    e5ba7672    f54016b9    21ddcdc9    b1252a9d    07b5194c        3a171ecb    c5c50484    e8b83407    9727dd16
```

经过get_slot_data.py处理后，得到如下数据, dense_feature中的值会merge在一起，对应net.py中的self._dense_data_var, '1:715353'表示net.py中的self._sparse_data_var[1] = 715353
```
click:0 dense_feature:0.05 dense_feature:0.00663349917081 dense_feature:0.05 dense_feature:0.0 dense_feature:0.02159375 dense_feature:0.008 dense_feature:0.15 dense_feature:0.04 dense_feature:0.362 dense_feature:0.1 dense_feature:0.2 dense_feature:0.0 dense_feature:0.04 1:715353 2:817085 3:851010 4:833725 5:286835 6:948614 7:881652 8:507110 9:27346 10:646986 11:643076 12:200960 13:18464 14:202774 15:532679 16:729573 17:342789 18:562805 19:880474 20:984402 21:666449 22:26235 23:700326 24:452909 25:884722 26:787527
```


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
python -m paddlerec.run -m paddlerec.models.rank.fibinet
```

## 复现论文&模型效果

用原论文的完整数据复现论文效果需要在config.py中修改batch_size=1000, thread_num=8, epoch_num=4

使用gpu单卡训练26h 测试auc:0.79

修改后运行有两种方案：
```
python setup.py install  # 安装到python  
python -m paddlerec.run -m paddlerec.models.rank.fibinet #运行 
```
第二种方案是修改config.py中'workspace'为config.py的目录位置，执行
```
python -m paddlerec.run -m /home/your/dir/config.yaml #调试模式 直接指定本地config的绝对路径
```

## 结果展示

样例数据训练结果展示：

```
Running SingleStartup.
W0623 12:03:35.130075   509 device_context.cc:237] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 9.2, Runtime API Version: 9.0
W0623 12:03:35.134771   509 device_context.cc:245] device: 0, cuDNN Version: 7.3.
Running SingleRunner.
batch: 100, AUC: [0.6449976], BATCH_AUC: [0.69029814]
batch: 200, AUC: [0.6769844], BATCH_AUC: [0.70255003]
batch: 300, AUC: [0.67131597], BATCH_AUC: [0.68954499]
batch: 400, AUC: [0.68129822], BATCH_AUC: [0.70892718]
batch: 500, AUC: [0.68242937], BATCH_AUC: [0.69269376]
batch: 600, AUC: [0.68741928], BATCH_AUC: [0.72034578]
...
batch: 1400, AUC: [0.84607023], BATCH_AUC: [0.93358024]
batch: 1500, AUC: [0.84796116], BATCH_AUC: [0.95302841]
batch: 1600, AUC: [0.84949111], BATCH_AUC: [0.92868531]
batch: 1700, AUC: [0.85113661], BATCH_AUC: [0.95452616]
batch: 1800, AUC: [0.85260467], BATCH_AUC: [0.92847032]
epoch 3 done, use time: 1618.1106688976288
```

样例数据预测结果展示

```
load persistables from increment_model/3
batch: 20, AUC: [0.85304064], BATCH_AUC: [0.94178556]
batch: 40, AUC: [0.85304544], BATCH_AUC: [0.95207907]
batch: 60, AUC: [0.85303907], BATCH_AUC: [0.94782551]
batch: 80, AUC: [0.85298773], BATCH_AUC: [0.93987691]
...
batch: 1780, AUC: [0.866046], BATCH_AUC: [0.96424594]
batch: 1800, AUC: [0.86633785], BATCH_AUC: [0.96900967]
batch: 1820, AUC: [0.86662365], BATCH_AUC: [0.96759972]
```

