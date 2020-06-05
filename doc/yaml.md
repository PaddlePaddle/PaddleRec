# PaddleRec yaml配置说明

## 全局变量

 |   名称    |  类型  |                 取值                  | 是否必须 |                      作用描述                      |
 | :-------: | :----: | :-----------------------------------: | :------: | :------------------------------------------------: |
 | workspace | string | 路径 / paddlerec.models.{方向}.{模型} |    是    |           指定model/reader/data所在位置            |
 |   mode    | string |              runner名称               |    是    |             指定当次运行使用哪个runner             |
 |   debug   |  bool  |             True / False              |    否    | 当dataset.mode=QueueDataset时，开启op耗时debug功能 |



## runner变量

|             名称              |     类型     |               取值                | 是否必须 |                    作用描述                     |
| :---------------------------: | :----------: | :-------------------------------: | :------: | :---------------------------------------------: |
|             name              |    string    |               任意                |    是    |                 指定runner名称                  |
|             class             |    string    | single_train(默认) / single_infer |    是    | 指定运行runner的类别（单机/分布式， 训练/预测） |
|            device             |    string    |          cpu(默认) / gpu          |    否    |                  程序执行设备                   |
|            epochs             |     int      |               >= 1                |    否    |                模型训练迭代轮数                 |
|        init_model_path        |    string    |               路径                |    否    |                 初始化模型地址                  |
|   save_checkpoint_interval    |     int      |               >= 1                |    否    |               Save参数的轮数间隔                |
|     save_checkpoint_path      |    string    |               路径                |    否    |                 Save参数的地址                  |
|    save_inference_interval    |     int      |               >= 1                |    否    |             Save预测模型的轮数间隔              |
|      save_inference_path      |    string    |               路径                |    否    |               Save预测模型的地址                |
| save_inference_feed_varnames  | list[string] |     组网中指定Variable的name      |    否    |             预测模型的入口变量name              |
| save_inference_fetch_varnames | list[string] |     组网中指定Variable的name      |    否    |             预测模型的出口变量name              |
|        print_interval         |     int      |               >= 1                |    否    |              训练指标打印batch间隔              |



## phase变量

|     名称     |  类型  |     取值     | 是否必须 |            作用描述             |
| :----------: | :----: | :----------: | :------: | :-----------------------------: |
|     name     | string |     任意     |    是    |          指定phase名称          |
|    model     | string | model.py路径 |    是    | 指定Model()所在的python文件地址 |
| dataset_name | string | dataset名称  |    是    |       指定使用哪个Reader        |
|  thread_num  |  int   |     >= 1     |    否    |         模型训练线程数          |

## dataset变量

|      名称      |  类型  |           取值            | 是否必须 |            作用描述            |
| :------------: | :----: | :-----------------------: | :------: | :----------------------------: |
|      name      | string |           任意            |    是    |        指定dataset名称         |
|      type      | string | DataLoader / QueueDataset |    是    |        指定数据读取方式        |
|   batch_size   |  int   |           >= 1            |    否    |       指定批训练样本数量       |
|   data_path    | string |           路径            |    是    |        指定数据来源地址        |
| data_converter | string |       reader.py路径       |    是    | 指定Reader()所在python文件地址 |
|  sparse_slots  | string |          string           |    否    |        指定稀疏参数选项        |
|  dense_slots   | string |          string           |    否    |        指定稠密参数选项        |

## hyper_parameters变量
|          名称           |  类型  |       取值       | 是否必须 |          作用描述           |
| :---------------------: | :----: | :--------------: | :------: | :-------------------------: |
|     optimizer.class     | string | SGD/Adam/Adagrad |    是    |       指定优化器类型        |
| optimizer.learning_rate | float  |       > 0        |    否    |         指定学习率          |
|           reg           | float  |       > 0        |    否    | L2正则化参数，只在SGD下生效 |
|         others          |   /    |        /         |    /     |   由各个模型组网独立指定    |
