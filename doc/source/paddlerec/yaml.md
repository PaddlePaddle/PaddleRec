# yaml配置说明

## 全局变量

 |   名称    |         类型          |                         取值                          | 是否必须 |                      作用描述                      |
 | :-------: | :-------------------: | :---------------------------------------------------: | :------: | :------------------------------------------------: |
 | workspace |        string         |      绝对路径 或 paddlerec.models.{方向}.{模型}       |    是    |           指定model/reader/data所在位置            |
 |   mode    | string / list[string] | string：单个runner的名称 / list：多个runner名称的列表 |    是    |             指定当次运行使用哪些runner             |
 |   debug   |         bool          |                     True / False                      |    否    | 当dataset.mode=QueueDataset时，开启op耗时debug功能 |



## runner变量

|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             name              |    string    |                           任意                            |    是    |                            指定runner名称                            |
|             class             |    string    | train(默认) / infer / local_cluster_train / cluster_train |    是    |           指定运行runner的类别（单机/分布式， 训练/预测）            |
|            device             |    string    |                      cpu(默认) / gpu                      |    否    |                             程序执行设备                             |
|          fleet_mode           |    string    |               ps(默认) / pslib / collective               |    否    |                            分布式运行模式                            |
|         selected_gpus         |    string    |                         "0"(默认)                         |    否    | 程序运行GPU卡号，若以"0,1"的方式指定多卡，则会默认启用collective模式 |
|          worker_num           |     int      |                          1(默认)                          |    否    |                     参数服务器模式下worker的数量                     |
|          server_num           |     int      |                          1(默认)                          |    否    |                     参数服务器模式下server的数量                     |
|      distribute_strategy      |    string    |              async(默认)/sync/half_async/geo              |    否    |                    参数服务器模式下训练模式的选择                    |
|            epochs             |     int      |                           >= 1                            |    否    |                           模型训练迭代轮数                           |
|            phases             | list[string] |                  由phase name组成的list                   |    否    |                  当前runner的训练过程列表，顺序执行                  |
|        init_model_path        |    string    |                           路径                            |    否    |                            初始化模型地址                            |
|   save_checkpoint_interval    |     int      |                           >= 1                            |    否    |                          Save参数的轮数间隔                          |
|     save_checkpoint_path      |    string    |                           路径                            |    否    |                            Save参数的地址                            |
|    save_step_interval    |     int      |                           >= 1                            |    否    |                        Step save参数的batch数间隔                        |
|      save_step_path      |    string    |                           路径                            |    否    |                           Step save参数的地址                          |
|    save_inference_interval    |     int      |                           >= 1                            |    否    |                        Save预测模型的轮数间隔                        |
|      save_inference_path      |    string    |                           路径                            |    否    |                          Save预测模型的地址                          |
| save_inference_feed_varnames  | list[string] |                 组网中指定Variable的name                  |    否    |                        预测模型的入口变量name                        |
| save_inference_fetch_varnames | list[string] |                 组网中指定Variable的name                  |    否    |                        预测模型的出口变量name                        |
|        print_interval         |     int      |                           >= 1                            |    否    |                        训练指标打印batch间隔                         |
|      instance_class_path      |    string    |                           路径                            |    否    |                     自定义instance流程实现的地址                     |
|      network_class_path       |    string    |                           路径                            |    否    |                     自定义network流程实现的地址                      |
|      startup_class_path       |    string    |                           路径                            |    否    |                     自定义startup流程实现的地址                      |
|       runner_class_path       |    string    |                           路径                            |    否    |                      自定义runner流程实现的地址                      |
|      terminal_class_path      |    string    |                           路径                            |    否    |                     自定义terminal流程实现的地址                     |
|  init_pretraining_model_path  |    string    |                           路径                            |    否    |自定义的startup流程中需要传入这个参数，finetune中需要加载的参数的地址 |
|  runner_result_dump_path  |    string    |                           路径                            |    否    | 运行中metrics的结果使用json.dump到文件的地址，若是在训练的runner中使用, 会自动加上epoch后缀 |




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


## PaddleRec backend.yaml配置说明

## 全局变量

 |     名称     |  类型  |      取值       | 是否必须 |                     作用描述                     |
 | :----------: | :----: | :-------------: | :------: | :----------------------------------------------: |
 |   backend    | string | paddlecloud/k8s |    是    | 使用PaddleCloud平台提交，还是在公有云K8S集群提交 |
 | cluster_type | string |     mpi/k8s     |    是    |        指定运行的计算集群： mpi 还是 k8s         |

 ## config

 |          名称          |  类型  |                  取值                   | 是否必须 |                                           作用描述                                           |
 | :--------------------: | :----: | :-------------------------------------: | :------: | :------------------------------------------------------------------------------------------: |
 |     paddle_version     | string | paddle官方版本号，如1.7.2/1.8.0/1.8.3等 |    否    |                           指定运行训练使用的Paddle版本，默认1.7.2                            |
 |      use_python3       |  int   |               0（默认）/1               |    否    |                                 指定是否使用python3进行训练                                  |
 |        fs_name         | string |             "afs://xxx.com"             |    是    |                                   hdfs/afs集群名称所需配置                                   |
 |         fs_ugi         | string |                "usr,pwd"                |    是    |                                   hdfs/afs集群密钥所需配置                                   |
 |      output_path       | string |            "/user/your/path"            |    否    |                                      任务输出的远程目录                                      |
 |    train_data_path     | string |            "/user/your/path"            |    是    | mpi集群下指定训练数据路径，paddlecloud会自动将数据分片并下载到工作目录的`./train_data`文件夹 |
 |     test_data_path     | string |            "/user/your/path"            |    否    |             mpi集群下指定测试数据路径，会自动下载到工作目录的`./test_data`文件夹             |
 |    thirdparty_path     | string |            "/user/your/path"            |    否    |           mpi集群下指定thirdparty路径，会自动下载到工作目录的`./thirdparty`文件夹            |
 | afs_remote_mount_point | string |            "/user/your/path"            |    是    |                  k8s集群下指定远程路径的地址，会挂载到工作目录的`./afs/下`                   |
 
 ### config.communicator

 |                       名称                       | 类型  |      取值      | 是否必须 |                        作用描述                        |
 | :----------------------------------------------: | :---: | :------------: | :------: | :----------------------------------------------------: |
 |       FLAGS_communicator_is_sgd_optimizer        |  int  |  0（默认）/1   |    否    | 异步分布式训练时的多线程的梯度融合方式是否使用SGD模式  |
 |        FLAGS_communicator_send_queue_size        |  int  | 线程数（默认） |    否    |               分布式训练时发送队列的大小               |
 |       FLAGS_communicator_max_merge_var_num       |  int  | 线程数（默认） |    否    |        分布式训练多线程梯度融合时，线程数的配置        |
 | FLAGS_communicator_max_send_grad_num_before_recv |  int  | 线程数（默认） |    否    | 分布式训练使用独立recv参数线程时，与send的步调配置超参 |
 |       FLAGS_communicator_thread_pool_size        |  int  |   32（默认）   |    否    |        分布式训练时，多线程发送参数的线程池大小        |
 |           FLAGS_communicator_fake_rpc            |  int  |  0（默认）/1   |    否    |              分布式训练时，选择不进行通信              |
 |              FLAGS_rpc_retry_times               |  int  |    3(默认)     |    否    |            分布式训练时，GRPC的失败重试次数            |


## submit

|     名称      |  类型  |            取值             | 是否必须 |                         作用描述                         |
| :-----------: | :----: | :-------------------------: | :------: | :------------------------------------------------------: |
|      ak       | string | PaddleCloud平台提供的ak密钥 |    是    |                   paddlecloud用户配置                    |
|      sk       | string | PaddleCloud平台提供的sk密钥 |    否    |                   paddlecloud用户配置                    |
|   priority    | string |    normal/high/very_high    |    否    |                        任务优先级                        |
|   job_name    | string |            任意             |    是    |                         任务名称                         |
|     group     | string |     计算资源所在组名称      |    是    |                          组名称                          |
|   start_cmd   | string |            任意             |    是    | 启动命令，默认`python -m paddlerec.run -m ./config.yaml` |
|     files     | string |            任意             |    是    |         随任务提交上传的文件，给出相对或绝对路径         |
|     nodes     |  int   |        >=1（默认1）         |    否    |                    mpi集群下的节点数                     |
| k8s_trainers  |  int   |        >=1（默认1）         |    否    |                 k8s集群下worker的节点数                  |
| k8s_cpu_cores |  int   |        >=1（默认1）         |    否    |                 k8s集群下worker的CPU核数                 |
| k8s_gpu_card  |  int   |        >=1（默认1）         |    否    |                 k8s集群下worker的GPU卡数                 |
|  k8s_ps_num   |  int   |        >=1（默认1）         |    否    |                 k8s集群下server的节点数                  |
| k8s_ps_cores  |  int   |        >=1（默认1）         |    否    |                 k8s集群下server的CPU核数                 |
