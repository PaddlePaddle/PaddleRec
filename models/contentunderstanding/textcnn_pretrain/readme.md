# 使用文本分类模型作为预训练模型对textcnn模型进行fine-tuning

以下是本例的简要目录结构及说明： 

```
├── data #样例数据
    ├── train
        ├── train.txt #训练数据样例
    ├── test
        ├── test.txt #测试数据样例
    ├── preprocess.py #数据处理程序
├── __init__.py
├── README.md #文档
├── model.py #模型文件
├── basemodel.py #预训练模型
├── config.yaml #配置文件
├── reader.py #读取程序
├── finetune_startup.py #加载参数
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
情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极。在本文中，我们提供了一个使用大规模的对文章数据进行多分类的textCNN模型（2个卷积核的cnn模型）作为预训练模型。本文会使用这个预训练模型对contentunderstanding目录下的textcnn模型（3个卷积核的cnn模型）进行fine-tuning。本文将预训练模型中的embedding层迁移到了contentunderstanding目录下的textcnn模型中，依然进行情感分析的二分类任务。最终获得了模型准确率上的基本持平以及更快速的收敛  
Yoon Kim在论文[EMNLP 2014][Convolutional neural networks for sentence classication](https://www.aclweb.org/anthology/D14-1181.pdf)提出了TextCNN并给出基本的结构。将卷积神经网络CNN应用到文本分类任务，利用多个不同size的kernel来提取句子中的关键信息（类似于多窗口大小的ngram），从而能够更好地捕捉局部相关性。模型的主体结构如图所示：  
<p align="center">
<img align="center" src="../../../doc/imgs/cnn-ckim2014.png">
<p>

## 数据准备
情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。情感类型分为积极、消极。情感倾向分析能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持。  
情感是人类的一种高级智能行为，为了识别文本的情感倾向，需要深入的语义建模。另外，不同领域（如餐饮、体育）在情感的表达各不相同，因而需要有大规模覆盖各个领域的数据进行模型训练。为此，我们通过基于深度学习的语义模型和大规模数据挖掘解决上述两个问题。效果上，我们和contentunderstanding目录下的textcnn模型一样基于开源情感倾向分类数据集ChnSentiCorp进行评测。  
您可以直接执行以下命令获取我们的预训练模型（basemodel.py，pretrain_model_params）以及对应的字典（word_dict.txt）：
```
wget https://paddlerec.bj.bcebos.com/textcnn_pretrain%2Fpretrain_model.tar.gz
tar -zxvf textcnn_pretrain%2Fpretrain_model.tar.gz
```
您可以直接执行以下命令下载我们分词完毕后的数据集,文件解压之后，senta_data目录下会存在训练数据（train.tsv）、开发集数据（dev.tsv）、测试集数据（test.tsv）以及对应的词典（word_dict.txt）：  
``` 
wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
tar -zxvf sentiment_classification-dataset-1.0.0.tar.gz
```
数据格式为一句中文的评价语句，和一个代表情感信息的标签。两者之间用/t分隔，中文的评价语句已经分词，词之间用空格分隔。  
```
15.4寸 笔记本 的 键盘 确实 爽 ， 基本 跟 台式机 差不多 了 ， 蛮 喜欢 数字 小 键盘 ， 输 数字 特 方便 ， 样子 也 很 美观 ， 做工 也 相当 不错    1
跟 心灵 鸡汤 没 什么 本质 区别 嘛 ， 至少 我 不 喜欢 这样 读 经典 ， 把 经典 都 解读 成 这样 有点 去 中国 化 的 味道 了 0
```

## 运行环境
PaddlePaddle>=1.7.2

python 2.7/3.5/3.6/3.7

PaddleRec >=0.1

os : windows/linux/macos

## 快速开始
本文需要下载模型的参数文件和finetune的数据集才可以体现出finetune的效果，所以暂不提供快速一键运行。若想体验finetune的效果，请按照下面【效果复现】模块的步骤依次执行。 

## 效果复现
在本模块，我们希望用户可以理解如何使用预训练模型来对自己的模型进行fine-tuning。
1. 确认您当前所在目录为PaddleRec/models/contentunderstanding/textcnn_pretrain

2. 下载并解压数据集，命令如下。解压后您可以看到出现senta_data目录
``` 
wget https://baidu-nlp.bj.bcebos.com/sentiment_classification-dataset-1.0.0.tar.gz
tar -zxvf sentiment_classification-dataset-1.0.0.tar.gz
```

3. 下载并解压预训练模型，命令如下。
```
wget https://paddlerec.bj.bcebos.com/textcnn_pretrain%2Fpretrain_model.tar.gz
tar -zxvf textcnn_pretrain%2Fpretrain_model.tar.gz
```

4. 本文提供了快速将数据集中的汉字数据处理为可训练格式数据的脚本。在您下载预训练模型后，将word_dict.txt复制到senta_data文件中。您在解压数据集后，将preprocess.py复制到senta_data文件中。
执行preprocess.py，即可将数据集中提供的dev.tsv，test.tsv，train.tsv按照词典提供的对应关系转化为可直接训练的txt文件.命令如下：
```
rm -f senta_data/word_dict.txt
cp pretrain_model/word_dict.txt senta_data
cp data/preprocess.py senta_data/
cd senta_data
python3 preprocess.py
mkdir train
mv train.txt train
mkdir test
mv test.txt  test
cd ..
```

5. 打开文件config.yaml,更改其中的参数
将workspace改为您当前的绝对路径。（可用pwd命令获取绝对路径）  


6. 执行命令，开始训练：
```
python -m paddlerec.run -m ./config.yaml
```

7. 运行结果：
```
PaddleRec: Runner infer_runner Begin
Executor Mode: infer
processor_register begin
Running SingleInstance.
Running SingleNetwork.
Running SingleInferStartup.
Running SingleInferRunner.
load persistables from increment/3
batch: 1, acc: [0.8828125], loss: [0.35940486]
batch: 2, acc: [0.91796875], loss: [0.24300358]
batch: 3, acc: [0.91015625], loss: [0.2490797]
Infer phase_infer of epoch increment/3 done, use time: 0.78388094902, global metrics: acc=[0.91015625], loss=[0.2490797]
PaddleRec Finish
```

## 进阶使用
在观察完model.py和config.yaml两个文件后，相信大家会发现和之前的模型相比有些改变。本章将详细解析这些改动，方便大家理解并灵活应用到自己的程序中.  
1.在model.py中，大家会发现在构建embedding层的时候，直接传参使用了basemodel.py中的embeding层。  
这是因为本文使用了预训练模型（basemodel.py）中embedding层，经过大量语料的训练后的embedding层中本身已经蕴含了大量的先验知识。而这些先验知识对于下游任务，尤其是小数据集来讲，是非常有帮助的。  

2.在config.yaml中，大家会发现在train_runner中多了startup_class_path和init_pretraining_model_path两个参数。  
参数startup_class_path的作用是自定义训练的流程。我们将在自定义的finetune_startup.py文件中将训练好的参数加载入模型当中。  
参数init_pretraining_model_path的作用就是指明加载参数的路径。若路径下的参数文件和模型中的var具有相同的名字，就会将参数加载进模型当中。
在您设置init_model_path参数时，程序会优先试图按您设置的路径热启动。当没有init_model_path参数，无法热启动时，程序会试图加载init_pretraining_model_path路径下的参数，进行finetune训练。  
只有在两者均为空的情况下，模型会冷启动从头开始训练。
若您希望进一步了解自定义流程的操作，可以参考以下内容：[如何添加自定义流程](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/trainer_develop.md#%E5%A6%82%E4%BD%95%E6%B7%BB%E5%8A%A0%E8%87%AA%E5%AE%9A%E4%B9%89%E6%B5%81%E7%A8%8B) 

3.在basemodel.py中，我们准备了embedding，multi_convs，full_connect三个模块供您在有需要时直接import使用。  
相关参数可以从本文提供的预训练模型下载链接里的pretrain_model/pretrain_model_params中找到。

## FAQ
