# GNN

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
	├── train
		├── convert_sample.txt
    ├── test
		├── sample.txt
    ├── dict
		├── word_count_dict.txt
        ├── word_id_dict.txt
├── preprocess.py # 数据预处理文件
├── __init__.py
├── README.md # 文档
├── model.py #模型文件
├── config.yaml #配置文件
├── data_prepare.sh #一键数据处理脚本
├── w2v_reader.py #训练数据reader
├── w2v_evaluate_reader.py # 预测数据reader
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
本例实现了skip-gram模式的word2vector模型，如下图中的skip-gram部分：
![](https://ai-studio-static-online.cdn.bcebos.com/bf0217bcb42e455284290a100670072989d432939b7e43e38b78eea6b60732c0)  
以每一个词为中心词X，然后在窗口内和临近的词Y组成样本对（X,Y）用于网络训练。在实际训练过程中还会根据自定义的负采样率生成负样本来加强训练的效果  
具体的训练思路如下：  
![](https://ai-studio-static-online.cdn.bcebos.com/ad45d6dfff2f4fa69c639a6b5d2bbe46c79ac67658464aa0a069a04eb2800cb6)

推荐用户参考[ IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/124377)教程获取更详细的信息。

本模型配置默认使用demo数据集，若进行精度验证，请参考[论文复现](#论文复现)部分。

本项目支持功能

训练：单机CPU、本地模拟参数服务器训练、增量训练，配置请参考 [启动训练](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/train.md)   

预测：单机CPU；配置请参考[PaddleRec 离线预测](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/predict.md) 

## 数据处理
本示例中数据处理共包含三步：
- Step1: 数据下载。
    ```
    # 全量训练集
    mkdir raw_data
    wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/1-billion-word-language-modeling-benchmark-r13output.tar
    tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar
    mv 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ raw_data/

    # 测试集
    wget --no-check-certificate https://paddlerec.bj.bcebos.com/word2vec/test_dir.tar
    tar xzvf test_dir.tar -C raw_data
    ```

- Step2: 训练据预处理。包含三步，第一步，根据英文语料生成词典，中文语料可以通过修改text_strip方法自定义处理方法。
    ```
    python preprocess.py --build_dict --build_dict_corpus_dir raw_data/training-monolingual.tokenized.shuffled --dict_path raw_data/word_count_dict.txt
    ```
    得到的词典格式为词<空格>词频，低频词用'UNK'表示，如下所示：
    ```
    the 1061396
    of 593677
    and 416629
    one 411764
    in 372201
    a 325873
    <UNK> 324608
    to 316376
    zero 264975
    nine 250430
    ```
    第二步，根据词典将文本转成id, 同时进行downsample，按照概率过滤常见词, 同时生成word和id映射的文件，文件名为词典+"word_to_id"。
    ```
    python preprocess.py --filter_corpus --dict_path raw_data/word_count_dict.txt --input_corpus_dir raw_data/training-monolingual.tokenized.shuffled --output_corpus_dir raw_data/convert_text8 --min_count 5 --downsample 0.001
    ```
    第三步，为更好地利用多线程进行训练加速，我们需要将训练文件分成多个子文件，默认拆分成1024个文件，文件保存在data/train目录下。
    ```
    rm -rf data/train/*
    python preprocess.py --data_resplit --input_corpus_dir=raw_data/convert_text8 --output_corpus_dir=data/train
    ```
- Step3: 数据整理。 将训练文件统一放在data/train目录下，测试文件统一放在data/test目录下，词典相关文件放在data/dict目录下。
    ```
    rm -rf data/test/*
    mv raw_data/data/test_dir/* data/test/

    mv raw_data/word_count_dict.txt data/dict/
    mv raw_data/word_id_dict.txt data/dict/ 
    rm -rf raw_data
    ```
方便起见， 我们提供了一键式数据处理脚本：
```
sh data_prepare.sh
```

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
  epochs: 5
  # device to run training or infer
  device: cpu
  save_checkpoint_interval: 1 # save model interval of epochs
  save_inference_interval: 1 # save inference
  save_checkpoint_path: "increment_w2v" # save checkpoint path
  save_inference_path: "inference_w2v" # save inference path
  save_inference_feed_varnames: [] # feed vars of save inference
  save_inference_fetch_varnames: [] # fetch vars of save inference
  init_model_path: "" # load model path
  print_interval: 1
  phases: [phase1]
```
### 单机预测

CPU环境

在config.yaml文件中设置好epochs、device等参数。

```
- name: single_cpu_infer
  class: infer
  # device to run training or infer
  device: cpu
  init_model_path: "increment_w2v" # load model path
  print_interval: 1
  phases: [phase2]
```

### 运行
```
python -m paddlerec.run -m paddlerec.models.recall.w2v
```

### 结果展示

样例数据训练结果展示：

```
Running SingleStartup.
Running SingleRunner.
W0813 11:36:16.129736 43843 build_strategy.cc:170] fusion_group is not enabled for Windows/MacOS now, and only effective when running with CUDA GPU.
batch: 1, LOSS: [3.618 3.684 3.698 3.653 3.736]
batch: 2, LOSS: [3.394 3.453 3.605 3.487 3.553]
batch: 3, LOSS: [3.411 3.402 3.444 3.387 3.357]
batch: 4, LOSS: [3.557 3.196 3.304 3.209 3.299]
batch: 5, LOSS: [3.217 3.141 3.168 3.114 3.315]
batch: 6, LOSS: [3.342 3.219 3.124 3.207 3.282]
batch: 7, LOSS: [3.19  3.207 3.136 3.322 3.164]
epoch 0 done, use time: 0.119026899338, global metrics: LOSS=[3.19  3.207 3.136 3.322 3.164]
...
epoch 4 done, use time: 0.097608089447, global metrics: LOSS=[2.734 2.66  2.763 2.804 2.809]
```
样例数据预测结果展示:
```
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from increment_w2v/4
batch: 1, acc: [1.]
batch: 2, acc: [1.]
batch: 3, acc: [1.]
Infer phase2 of epoch 4 done, use time: 4.89376211166, global metrics: acc=[1.]
...
Infer phase2 of epoch 3 done, use time: 4.43099021912, global metrics: acc=[1.]
```

## 论文复现

用原论文的完整数据复现论文效果需要在config.yaml修改超参：
- batch_size: 修改config.yaml中dataset_train数据集的batch_size为100。
- epochs: 修改config.yaml中runner的epochs为5。

使用cpu训练 5轮 测试Recall@20:0.540

修改后运行方案：修改config.yaml中的'workspace'为config.yaml的目录位置，执行
```
python -m paddlerec.run -m /home/your/dir/config.yaml #调试模式 直接指定本地config的绝对路径
```

## 进阶使用

## FAQ
