
# Benchmark 运行指南

## 数据处理
为和样例数据路径区分，全量训练数据、测试数据、词表文件会依次保存在data/all_train, data/all_test, data/all_dict文件夹中。
```
mkdir -p data/all_dict
mkdir -p data/all_train
mkdir -p data/all_test
```
本示例中全量数据处理共包含三步：
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
    mv raw_data/data/test_dir/* data/all_test/
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
    第三步，为更好地利用多线程进行训练加速，我们需要将训练文件分成多个子文件，默认拆分成1024个文件。
    ```
    python preprocess.py --data_resplit --input_corpus_dir=raw_data/convert_text8 --output_corpus_dir=data/all_train
    ```
- Step3: 路径整理。
    ```
    mv raw_data/word_count_dict.txt data/all_dict/
    mv raw_data/word_count_dict.txt_word_to_id_ data/all_dict/word_id_dict.txt
    rm -rf raw_data
    ```
方便起见， 我们提供了一键式数据处理脚本：
脚本位于 `paddlerec/models/recall/word2vec/data_prepare.sh`, 执行以下命令下载及处理数据

    ```
    cp ../../models/recall/word2vec/data_prepare.sh ./
    cp ../../models/recall/word2vec/preprocess.py ./
    sh data_prepare.sh
    ```

## 单机训练
在数据处理完成后，根据需要可修改`config.py`中的各项配置（文件中的默认值为后文Benchmark的数据所使用配置）

执行以下命令进行单机训练:

```shell
python -u train.py 
```

## 分布式训练
在数据处理完成后，根据需要可修改`config.py`中的各项配置（文件中的默认值为后文Benchmark的数据所使用配置）

执行以下命令进行本地模拟分布式训练:

```shell
fleetrun --worker_num=1 --server_num=1 train.py
```

在真实集群中，在各台机器上，分别执行以下命令进行分布式训练：

```shell
fleetrun --worker_ips="ip1:port1,ip2:port2" --server_ips="ip3:port3,ip4:port4" train.py
```


## Benchmark数据

在以下机器配置： 
> 占位

docker镜像：
> 占位

我们测得单机性能：
> 占位

集群配置为：
> 占位

测得分布式性能：
> 占位
