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
2. 为了记录特征展现(show)和点击(click)的次数，需要在网络中定义两个变量，指明特征是否展现和点击，取值均为0或者1，sparse_embedding中通过entry参数传入一个ShowClickEntry，指明这两个变量(show和click)的名字。
```python
# net.py
# 构造ShowClickEntry，指明展现和点击对应的变量名
self.entry = paddle.distributed.ShowClickEntry("show", "click")
emb = paddle.static.nn.sparse_embedding(
    input=s_input,
    size=[self.dict_dim, self.emb_dim],
    padding_idx=0,
    entry=self.entry,   # 在sparse_embedding中传入entry
    param_attr=paddle.ParamAttr(name="embedding"))

# static_model.py
# 构造show/click对应的data，变量名需要与entry中的名称一致
show = paddle.static.data(
    name="show", shape=[None, 1], dtype="int64", lod_level=1)
label = paddle.static.data(
    name="click", shape=[None, 1], dtype="int64", lod_level=1)
```

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
目前流式训练支持保存几种格式的模型：
### 全量模型
全量模型（checkpoint）用于流式训练的热启，具体目录为model_save_path/{$day}/{$pass_id}。  
其中model_save_path为config_online.yaml中的配置，day对应8位日期，pass_id对应流式训练过程中的第几个pass。  
目录下存在两个文件夹，其中000为模型中的dense参数，001为模型中的sparse参数。  
### batch model
与checkpoint模型类似，一般在每天数据训练结束后保存，保存前调用shrink函数删除掉长久不见的sparse特征，节省空间。
### base/delta model
base/delta模型一般用于线上预测，与全量模型相比，在保存过程中去掉了部分出现频率不高的特征，降低模型保存的磁盘占用及耗时。  
这两个模型一般指sparse_embedding中的参数，因此需要搭配dnn_plugin（模型文件和dense参数文件）才能实现线上完整预测。  
base模型具体保存路径为model_save_path/{$day}/base，每天数据训练结束后保存，保存前调用shrink函数。  
delta模型具体保存路径为model_save_path/{$day}/delta_{$pass_id}，每一个delta模型都是在上一个base/delta模型基础上进行保存的增量模型。  

## 高级功能
为进一步提升模型效果，降低存储空间，提供了一系列高级功能，下面逐一进行介绍相关的功能和配置。  
具体配置详情可参考config_online.yaml中的table_parameters部分。  
为使用高级功能，需要配置相应的table及accessor：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             table_class              |    string    |     MemorySparseTable           |    是    |        存储embedding的table名称     |
|             accessor_class              |    string    |              CtrCommonAccessor          |    是    |       获取embedding的accessor名称       |
### 特征频次score计算
server端会根据特征的show和click计算一个频次得分，用于判断该特征embedding是否可以扩展、保存等，具体涉及到的配置如下：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             nonclk_coeff              |    float    |                           任意                            |    是    |                            特征展现但未点击对应系数                            |
|             click_coeff              |    float    |                           任意                            |    是    |                            特征点击对应系数                            |

具体频次score计算公式如下：  
score = click_coeff * click + noclick_coeff * (click - show)
### 特征embedding扩展
特征embedding初始情况下，只会生成一维embedding，其余维度均为0，当特征的频次score大于等于扩展阈值时，才会扩展出剩余维度，具体涉及到的配置如下：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             embedx_dim              |    int    |                           任意                            |    是    |                            特征embedding扩展维度                            |
|             embedx_threshold              |    int    |                           任意                            |    是    |                            特征embedding扩展阈值                            |
|             fea_dim              |    int    |                           任意                            |    是    |                            特征embedding总维度                            |

需要注意的是：  
1. 特征embedding的实际维度为1 + embedx_dim，即一维初始embedding + 扩展embedding。
2. 特征总维度包括show和click，因此fea_dim = embedx_dim + 3。
### 特征embedding保存
为降低模型保存的磁盘占用及耗时，在保存base/delta模型时，可以去掉部分出现频率不高的特征，具体涉及到的配置如下：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             base_threshold              |    float    |                           任意                            |    是    |       特征频次score大于等于该阈值才会在base模型中保存                            |
|             delta_threshold              |    float    |                           任意                      |    是    |   从上一个delta模型到当前delta模型，<br>特征频次score大于等于该阈值才会在delta模型中保存        |
|             delta_keep_days              |    int    |                           任意                        |    是    |   特征未出现天数小于等于该阈值才会在delta模型中保存               |
|             converter              |    string    |              任意                        |    否    |   base/delta模型转换器（对接线上推理KV存储）            |
|             deconverter              |    string    |              任意                        |    否    |   base/delta模型解压器               |
### 特征embedding淘汰
一般每天的数据训练完成后，会调用shrink函数删除掉一些长久不出现或者出现频率极低的特征，具体涉及到的配置如下：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             show_click_decay_rate              |    float    |                           任意                        |    是    |   调用shrink函数时，show和click会根据该配置进行衰减               |
|             delete_threshold              |    float    |                           任意                            |    是    |       特征频次score小于该阈值时，删除该特征                 |
|             delete_after_unseen_days              |    int    |                           任意                            |    是    |       特征未出现天数大于该阈值时，删除该特征                 |
### 参数优化算法
稀疏参数(sparse_embedding)优化算法配置，分为一维embedding的优化算法(embed_sgd_param)和扩展embedding的优化算法(embedx_sgd_param)：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             name              |    string    |    SparseAdaGradSGDRule<br>SparseNaiveSGDRule<br>SparseAdamSGDRule<br>StdAdaGradSGDRule      |    是    |       优化算法名称                 |
|             learning_rate              |    float    |    任意                  |    是    |       学习率                 |
|             initial_g2sum              |    float    |    任意                  |    是    |       g2sum初始值                 |
|             initial_range              |    float    |    任意                  |    是    |       embedding初始化范围[-initial_range, initial_range]          |
|             weight_bounds              |    list(float)    |    任意                  |    是    |    embedding在训练过程中的范围        |

稠密参数优化算法配置：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             adam_d2sum              |    bool    |    任意                        |    是    |       是否使用新的稠密参数优化算法                 |
