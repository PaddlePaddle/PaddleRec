# 流式训练
推荐系统在服务的过程中，会不断产生可用于训练CTR模型的日志数据，流式训练是指数据不是一次性放入训练系统中，而是随着时间流式地加入到训练过程中去。每接收一个分片的数据，模型会对它进行预测，并利用该分片数据增量训练模型，同时按一定的频率保存全量或增量模型。  
本教程以[slot_dnn](../models/rank/slot_dnn/README.md)模型使用demo数据为例进行介绍。  

## 配置
流式训练配置参见models/rank/slot_dnn/config_online.yaml，新增配置及作用如下：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             split_interval              |    int    |                           任意                            |    是    |                            数据落盘分片间隔时间（分钟）                            |
|             split_per_pass              |    int    |                           任意                            |    是    |                            训练一个pass包含多少个分片的数据                            |
|             start_day              |    string    |                           任意                            |    是    |                            训练开始的日期（例：20190720）                            |
|             end_day              |    string    |                           任意                            |    是    |                            训练结束的日期（例：20190720）                            |
|             checkpoint_per_pass              |    int    |                           任意                            |    是    |                            训练多少个pass保存一个模型（模型保存的频率）                            |
|             data_donefile              |    string    |                           任意                            |    否    |                            用于探测当前分片数据是否落盘完成的标识文件                            |
|             data_sleep_second              |    int    |                           任意                            |    否    |                            当前分片数据尚未完成的等待时间                            |

## 数据
### 数据落盘目录格式
在train_data_dir目录下，再建立两层目录，第一层目录对应训练数据的日期（8位），第二层目录对应训练数据的具体时间（4位，前两位为小时，后两位为分钟），并且需要与配置文件中的split_interval配置对应。  
例如：train_data_dir配置为“data”目录，split_interval配置为5，则具体的目录结构如下：  
```txt
├── data
    ├── 20190720              # 训练数据的日期
        ├── 0000              # 训练数据的时间（第1个分片）
            ├── data_part1    # 具体的训练数据文件
            ├── ......    
        ├── 0005              # 训练数据的时间（第2个分片）
            ├── data_part1    # 具体的训练数据文件
            ├── ......
        ├── 0010              # 训练数据的时间（第3个分片）
            ├── data_part1    # 具体的训练数据文件
            ├── ......
        ├── ......
        ├── 2355              # 训练数据的时间（该日期下最后1个分片）
            ├── data_part1    # 具体的训练数据文件
            ├── ......
```
### 数据等待方式
开启配置中的data_donefile后，当数据目录中存在data_donefile配置对应的文件（一般是一个空文件）时，才会对该目录下的数据执行后续操作，否则，等待data_sleep_second时间后，重新探测是否存在data_donefile文件。

## 模型
流式训练采用静态图参数服务器方式训练，在组网时需要注意几点：
1. embedding层需使用paddle.static.nn.sparse_embedding，其中size参数的第一维可指定任意值，第二维为embedding向量的维度。  
2. 在组网中增加inference_feed_vars、inference_target_var两个变量的赋值，指明inference_model的输入和输出，供在线推理使用。
3. 在组网中增加all_vars变量的赋值，可用于在线离线一致性检查。
4. 如果希望在训练过程中dump出组网中的变量和网络参数（主要用于训练中的调试和异常检查），请赋值train_dump_fields和train_dump_params；如果希望在预测过程中dump出组网中的变量（主要用于线上预测所需特征的离线灌库），请赋值infer_dump_fields。

## 训练
请在models/rank/slot_dnn目录下执行如下命令，启动流式训练。  
```bash
fleetrun --server_num=1 --worker_num=1 ../../../tools/static_ps_online_trainer.py -m config_online.yaml
```
启动后，可以在该目录下的log/workerlog.0文件中查看训练日志。  
正确的训练过程应该包含以下几个部分：  
1. 参数初始化：打印config_online.yaml中配置的参数。  
2. 获取之前已经训练好的模型并加载模型，如果之前没有保存模型，则跳过加载模型这一步。  
3. 循环训练每个pass的数据，其中包括获取训练数据（建立训练数据处理pipe）；利用上个pass的模型预测当前pass的数据，并获取预测AUC；训练当前pass的数据，并获取训练AUC。  
4. 保存模型：根据checkpoint_per_pass配置，在固定pass数据训练完成之后，保存模型。  

## 模型
目前流式训练支持保存两种格式的模型。  
### 全量模型
全量模型（checkpoint）用于流式训练的热启，具体目录为model_save_path/{$day}/{$pass_id}。
其中model_save_path为config_online.yaml中的配置，day对应8位日期，pass_id对应流式训练过程中的第几个pass。  
目录下的embedding.shard目录为sparse特征对应的embedding，其中.txt文件为具体的embedding值和优化方法需要的统计量，.meta文件指明.txt文件的具体schema。  
目录下的其他文件为dense参数，文件名即为这些参数在组网中对应的var_name。  
### inference_model
用于在线推理的模型，保存于model_save_path/day/inference_model_{$pass_id}中，分为model、sparse、dense三个部分。  
其中sparse和dense参数与checkpoint模型类似，多出一个名为“__model__”的文件，保存的是在线服务使用的组网（可能经过裁剪），线上服务可以直接加载。  
