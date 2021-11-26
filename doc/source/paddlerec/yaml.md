# PaddleRec config.yaml配置说明

目前支持runner和hyper_parameters的读取。

## runner变量

|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             train_data_dir              |    string    |                           任意                            |    是    |                            指定训练数据目录                            |
|             train_reader_path              |    string    |                           任意                            |    是    |                指定训练时用的Reader()所在python文件地址                |
|             train_batch_size            |    int    |                           >= 1                           |    是    |                       指定train阶段的批训练样本数量                        |
|             model_save_path            |    string    |                           任意                           |    是    |                     指定train阶段完成后Save参数的地址                      |
|             test_data_dir              |    string    |                           任意                            |    是    |                            指定测试数据目录                            |
|             infer_reader_path              |    string    |                           任意                            |    是    |                指定测试时用的Reader()所在python文件地址                |
|             infer_batch_size            |    int    |                           >= 1                           |    是    |                       指定infer阶段的批训练样本数量                        |
|             infer_load_path            |    string    |                           任意                           |    是    |                     指定infer阶段开始时初始化模型地址                     |
|             infer_start_epoch            |    int    |                           >= 0                           |    是    |    初始化模型时从第几个epoch保留的参数开始加载（从0开始计数，包括本次）    |
|             infer_end_epoch            |    int    |                           >= 0                           |    是    |    初始化模型时到第几个epoch保留的参数停止加载（从0开始技术，不包括本次）    |
|             use_gpu            |    bool    |                           True/False                           |    是    |                       指定是否使用gpu，若为False则默认使用cpu                        |
|             epochs            |    int    |                           >= 1                           |    是    |                       指定train阶段需要训练几个epoch                        |
|             print_interval            |    int    |                           >= 1                           |    是    |                       训练指标打印batch间隔                        |
|             use_auc            |    bool    |                           True/False                           |    否    |                       在每个epoch开始时重置auc指标的值                        |
|             use_visual            |    bool    |                           True/False                           |    否    |                       开启模型训练的可视化功能，开启时需要安装visualDL                        |
|             use_inference            |    bool    |                           True/False                           |    否    |                     是否使用save_inference_model接口保存                      |
|             save_inference_feed_varnames         |    list[string]    |                      组网中指定Variable的name                      |    否    |                     预测模型的入口变量name                     |
|             save_inference_fetch_varnames         |    list[string]    |                      组网中指定Variable的name                      |    否    |                     预测模型的出口变量name                     |
|             use_fleet         |    bool    |                      True/False                      |    否    |                     指定是否使用分布式运行单机多卡或多机多卡                     |
|             reader_type         |    string    |                      QueueDataset/DataLoader/CustomizeDataLoader                    |    否    |                     指定使用的reader类型                     |
|             model_init_path         |    string    |                      任意                      |    否    |                     指定是否使用热启动，在训练初期加载初始化模型                     |


## hyper_parameters变量
|          名称           |  类型  |       取值       | 是否必须 |          作用描述           |
| :---------------------: | :----: | :--------------: | :------: | :-------------------------: |
|     optimizer.class     | string | SGD/Adam/Adagrad |    是    |       指定优化器类型        |
| optimizer.learning_rate | float  |       > 0        |    否    |         指定学习率          |
|           reg           | float  |       > 0        |    否    | L2正则化参数，只在SGD下生效 |
|         others          |   /    |        /         |    /     |   由各个模型组网独立指定    |
