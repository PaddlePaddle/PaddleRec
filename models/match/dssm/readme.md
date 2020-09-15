# DSSM文本匹配模型

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── train
        ├── train.txt #训练数据样例
    ├── test
        ├── test.txt #测试数据样例
    ├── preprocess.py #数据处理程序
    ├── data_process #数据一键处理脚本
├── __init__.py
├── README.md #文档
├── model.py #模型文件
├── config.yaml #配置文件
├── synthetic_reader.py #读取训练集的程序
├── synthetic_evaluate_reader.py #读取测试集的程序
├── transform.py #将数据整理成合适的格式方便计算指标
├── run.sh #全量数据集中的训练脚本，从训练到预测并计算指标

```

注：在阅读该示例前，建议您先了解以下内容：

[paddlerec入门教程](https://github.com/PaddlePaddle/PaddleRec/blob/master/README.md)

## 内容

- [模型简介](#模型简介)
- [数据准备](#数据准备)
- [运行环境](#运行环境)
- [快速开始](#快速开始)
- [效果复现](#效果复现)
- [进阶使用](#进阶使用)
- [FAQ](#FAQ)


## 模型简介
DSSM是Deep Structured Semantic Model的缩写，即我们通常说的基于深度网络的语义模型，其核心思想是将query和doc映射到到共同维度的语义空间中，通过最大化query和doc语义向量之间的余弦相似度，从而训练得到隐含语义模型，达到检索的目的。DSSM有很广泛的应用，比如：搜索引擎检索，广告相关性，问答系统，机器翻译等。    
DSSM 的输入采用 BOW（Bag of words）的方式，相当于把字向量的位置信息抛弃了，整个句子里的词都放在一个袋子里了。将一个句子用这种方式转化为一个向量输入DNN中。  
Query 和 Doc 的语义相似性可以用这两个向量的 cosine 距离表示，然后通过softmax 函数选出与Query语义最相似的样本 Doc 。  

模型的具体细节可以阅读论文[DSSM](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf):
<p align="center">
<img align="center" src="../../../doc/imgs/dssm.png">
<p>

## 数据准备
BQ是一个智能客服中文问句匹配数据集，该数据集是自动问答系统语料，共有120,000对句子对，并标注了句子对相似度值。数据中存在错别字、语法不规范等问题，但更加贴近工业场景。执行以下命令可以获取上述数据集。
```
wget https://paddlerec.bj.bcebos.com/dssm%2Fbq.tar.gz
tar xzf dssm%2Fbq.tar.gz
rm -f dssm%2Fbq.tar.gz
```
数据集样例：
```
请问一天是否都是限定只能转入或转出都是五万。    微众多少可以赎回短期理财        0
微粒咨询电话号码多少    你们的人工客服电话是多少        1
已经在银行换了新预留号码。      我现在换了电话号码，这个需要更换吗      1
每个字段以tab键分隔，第1，2列表示两个文本。第3列表示类别（0或1，0表示两个文本不相似，1表示两个文本相似）。
```
## 运行环境
PaddlePaddle>=1.7.2

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos

## 快速开始
本文提供了样例数据可以供您快速体验，在paddlerec目录下执行下面的命令即可快速启动训练： 

```
python -m paddlerec.run -m models/match/dssm/config.yaml
```   

输出结果示例：
```
PaddleRec: Runner train_runner Begin
Executor Mode: train
processor_register begin
Running SingleInstance.
Running SingleNetwork.
file_list : ['models/match/dssm/data/train/train.txt']
Running SingleStartup.
Running SingleRunner.
!!! The CPU_NUM is not specified, you should set CPU_NUM in the environment variable list.
CPU_NUM indicates that how many CPUPlace are used in the current task.
And if this parameter are set as N (equal to the number of physical CPU core) the program may be faster.
export CPU_NUM=32 # for example, set CPU_NUM as number of physical CPU core which is 32.
!!! The default number of CPU_NUM=1.
I0821 06:56:26.224299 31061 parallel_executor.cc:440] The Program will be executed on CPU using ParallelExecutor, 1 cards are used, so 1 programs are executed in parallel.
I0821 06:56:26.231163 31061 build_strategy.cc:365] SeqOnlyAllReduceOps:0, num_trainers:1
I0821 06:56:26.237023 31061 parallel_executor.cc:307] Inplace strategy is enabled, when build_strategy.enable_inplace = True
I0821 06:56:26.240788 31061 parallel_executor.cc:375] Garbage collection strategy is enabled, when FLAGS_eager_delete_tensor_gb = 0
batch: 2, LOSS: [4.538238]
batch: 4, LOSS: [4.16424]
batch: 6, LOSS: [3.8121371]
batch: 8, LOSS: [3.4250507]
batch: 10, LOSS: [3.2285979]
batch: 12, LOSS: [3.2116117]
batch: 14, LOSS: [3.1406002]
epoch 0 done, use time: 0.357971906662, global metrics: LOSS=[3.0968776]
batch: 2, LOSS: [2.6843479]
batch: 4, LOSS: [2.546976]
batch: 6, LOSS: [2.4103594]
batch: 8, LOSS: [2.301374]
batch: 10, LOSS: [2.264183]
batch: 12, LOSS: [2.315862]
batch: 14, LOSS: [2.3409634]
epoch 1 done, use time: 0.22123003006, global metrics: LOSS=[2.344321]
batch: 2, LOSS: [2.0882485]
batch: 4, LOSS: [2.006743]
batch: 6, LOSS: [1.9231766]
batch: 8, LOSS: [1.8850241]
batch: 10, LOSS: [1.8829436]
batch: 12, LOSS: [1.9336565]
batch: 14, LOSS: [1.9784685]
epoch 2 done, use time: 0.212922096252, global metrics: LOSS=[1.9934461]
PaddleRec Finish
```
## 效果复现
为了方便使用者能够快速的跑通每一个模型，我们在每个模型下都提供了样例数据。如果需要复现readme中的效果,请按如下步骤依次操作即可。  
1. 确认您当前所在目录为PaddleRec/models/match/dssm
2. 在data目录下载并解压数据集，命令如下：  
``` 
cd data
wget https://paddlerec.bj.bcebos.com/dssm%2Fbq.tar.gz
tar xzf dssm%2Fbq.tar.gz
rm -f dssm%2Fbq.tar.gz
```
3. 本文提供了快速将数据集中的汉字数据处理为可训练格式数据的脚本，您在解压数据集后，可以看见目录中存在一个名为bq的目录。将其中的train.txt文件移动到data目录下，然后可以在python3环境下运行我们提供的preprocess.py文件。即可生成可以直接用于训练的数据目录test.txt,train.txt和label.txt。将其放入train和test目录下以备训练时调用。生成时间较长，请耐心等待。命令如下：
```
mv bq/train.txt ./raw_data.txt
python3 preprocess.py
mkdir big_train
mv train.txt ./big_train
mkdir big_test
mv test.txt ./big_test
cd ..
```
也可以使用我们提供的一键数据处理脚本data_process.sh
```
sh data_process.sh
```
经过预处理的格式：  
训练集为三个稀疏的BOW方式的向量：query,pos,neg  
测试集为两个稀疏的BOW方式的向量：query,pos  
label.txt中对应的测试集中的标签

4. 退回dssm目录中，打开文件config.yaml,更改其中的参数  

将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  
将dataset_train中的batch_size从8改为128
将hyper_parameters中的slice_end从8改为128.当您需要改变batchsize的时候，这个参数也需要随之变化
将dataset_train中的data_path改为{workspace}/data/big_train
将dataset_infer中的data_path改为{workspace}/data/big_test
将hyper_parameters中的trigram_d改为6327

5.  执行脚本，开始训练.脚本会运行python -m paddlerec.run -m ./config.yaml启动训练，并将结果输出到result文件中。然后启动transform.py整合数据，最后计算出正逆序指标：
```
sh run.sh
```

输出结果示例：
```
................run.................
8989
pnr:2.75621659307
query_num:1369
pair_num:16240 , 16240
equal_num:77
正序率: 0.733774670544
pos_num: 11860 , neg_num: 4303
```

## 进阶使用
  
## FAQ
