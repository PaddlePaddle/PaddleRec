# GNN

## 快速开始
PaddleRec中每个内置模型都配备了对应的样例数据，用户可基于该数据集快速对模型、环境进行验证，从而降低后续的调试成本。在内置数据集上进行训练的命令为：
```
python -m paddlerec.run -m paddlerec.models.recall.gnn 
```

## 数据处理
- Step1: 原始数据数据集下载，本示例提供了两个开源数据集：DIGINETICA和Yoochoose，可选其中任意一个训练本模型。
    ```
    cd data && python download.py diginetica     # or yoochoose
    ```
    > [Yoochooses](https://2015.recsyschallenge.com/challenge.html)数据集来源于RecSys Challenge 2015，原始数据包含如下字段：
    1. Session ID – the id of the session. In one session there are one or many clicks.
    2. Timestamp – the time when the click occurred.
    3. Item ID – the unique identifier of the item.
    4. Category – the category of the item.

    > [DIGINETICA](https://competitions.codalab.org/competitions/11161#learn_the_details-data2)数据集来源于CIKM Cup 2016 _Personalized E-Commerce Search Challenge_项目。原始数据包含如下字段：
    1. sessionId - the id of the session. In one session there are one or many clicks.
    2. userId - the id of the user, with anonymized user ids.
    3. itemId - the unique identifier of the item.
    4. timeframe - time since the first query in a session, in milliseconds.
    5. eventdate - calendar date.

- Step2: 数据预处理
    ```
    cd data && python preprocess.py --dataset diginetica   # or yoochoose
    ```
    1. 以session_id为key合并原始数据集，得到每个session的日期，及顺序点击列表。
    2. 过滤掉长度为1的session；过滤掉点击次数小于5的items。
    3. 训练集、测试集划分。原始数据集里最新日期七天内的作为测试集，更早之前的数据作为测试集。

- Step3: 数据整理。 将训练文件统一放在data/train目录下，测试文件统一放在data/test目录下。
    ```
    cat data/diginetica/train.txt | wc -l >> data/config.txt    # or yoochoose1_4 or yoochoose1_64
    rm -rf data/train/*
    rm -rf data/test/*
    mv data/diginetica/train.txt data/train
    mv data/diginetica/test.txt data/test
    ```
数据处理完成后，data/train目录存放训练数据，data/test目录下存放测试数据，data/config.txt中存放数据统计信息，用以配置模型超参。

方便起见， 我们提供了一键式数据处理脚本：
```
sh data_prepare.sh diginetica      # or yoochoose1_4 or yoochoose1_64
```

## 实验配置

为在真实数据中复现论文中的效果，你还需要完成如下几步，PaddleRec所有配置均通过修改模型目录下的config.yaml文件完成：

1. 真实数据配置。config.yaml中数据集相关配置见`dataset`字段，数据路径通过`data_path`进行配置。用户可以直接将workspace修改为当前项目目录的绝对路径完成设置。
2. 超参配置。
    - batch_size: 修改config.yaml中dataset_train数据集的batch_size为100。
    - epochs: 修改config.yaml中runner的epochs为5。
    - sparse_feature_number: 不同训练数据集(diginetica or yoochoose)配置不一致，diginetica数据集配置为43098，yoochoose数据集配置为37484。具体见数据处理后得到的data/config.txt文件中第一行。
    - corpus_size: 不同训练数据集配置不一致，diginetica数据集配置为719470，yoochoose数据集配置为5917745。具体见数据处理后得到的data/config.txt文件中第二行。

## 训练
在完成[实验配置](##实验配置)后，执行如下命令完成训练：
```
python -m paddlerec.run -m ./config.yaml
```

## 测试
开始测试前，你需要完成如下几步配置：
1. 修改config.yaml中的mode，为infer_runner。
2. 修改config.yaml中的phase，为phase_infer，需按提示注释掉phase_trainer。
3. 修改config.yaml中dataset_infer数据集的batch_size为100。

完成上面两步配置后，执行如下命令完成测试：
```
python -m paddlerec.run -m ./config.yaml
```
