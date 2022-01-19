# PaddleRec小工具

本目录下包含了paddlerec的各种训练模式的核心代码，以及经常用到的一些工具脚本。 

## 工具简介

 |           程序名称            |                       实现的功能                    |                 支持的环境                 |            使用命令         |
 | :---------------------------: | :-----------------------------------------------: | :-------------------------------: |:-------------------------------: |
 |  trainer.py  |  trainer.py是使用动态图进行训练的相关代码，主要实现了动态图通用的训练流程  |  可以支持在windows/linux/macos环境下从任意目录通过相对路径启动。支持使用cpu/gpu运行  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：python -u tools/trainer.py -m models/rank/dnn/config.yaml  |
 |  infer.py  |  infer.py是使用动态图进行预测的相关代码，主要实现了动态图通用的预测流程  |  可以支持在windows/linux/macos环境下从任意目录通过相对路径启动。支持使用cpu/gpu运行  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：python -u tools/infer.py -m models/rank/dnn/config.yaml  |
 |  static_train.py  |  static_train.py是使用静态图进行训练的相关代码，主要实现了静态图通用的训练流程  |  可以支持在windows/linux/macos环境下从任意目录通过相对路径启动。支持使用cpu/gpu运行  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：python -u tools/static_trainer.py -m models/rank/dnn/config.yaml  |
 |  static_infer.py  |  static_infer.py是使用静态图进行预测的相关代码，主要实现了动态图通用的预测流程  |  可以支持在windows/linux/macos环境下从任意目录通过相对路径启动。支持使用cpu/gpu运行  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：python -u tools/static_infer.py -m models/rank/dnn/config.yaml  |
 |  static_ps_trainer.py  | static_ps_trainer.py是基于参数服务器模式(ParameterServer)的分布式训练相关代码，目前仅支持使用静态图的方式训练  |  可以支持在linux环境下从任意目录通过相对路径启动。  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：fleetrun --worker_num=1 --server_num=1 tools/static_ps_trainer.py -m models/rank/dnn/config.yaml  |
 |  static_gpubox_trainer.py  | static_gpubox_trainer.py是基于异构参数服务器纯GPU模式(ParameterServer)的分布式训练相关代码，目前仅支持使用静态图的方式训练  |  可以支持在linux环境下从任意目录通过相对路径启动。  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：sh tools/run_gpubox.sh  。详细可参考[GPUBOX原理与使用](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/performance/heterps.html) |
 |  cal_pos_neg.py  |  输入一个文件，文件中包含以"tab"分割的查询内容(query)，模型计算正负例的相似度(sim)和真实标签(label)，计算正逆序比(正序率)的脚本  |  可以在windows/linux/macos环境下从任意目录通过相对路径启动  |  支持在任意目录下运行，以dssm模型为示例，在dssm模型目录中运行的命令为：python ../../../tools/cal_pos_neg.py pair.txt  |
 |  feature_importance.py  |  模型特征重要性衡量工具  |  可以在linux环境下从任意目录通过相对路径启动  |  具体用法见[特征重要性](../doc/feature_importance.md)  |
 |  paddle_infer.py  |  使用python版本的预测库进行推理  |  可以支持在linux环境下启动  |  具体用法见[python端预测库推理](../doc/inference.md)  |
 |  rec_client.py  |  在服务器端启动serving服务成功后，部署客户端  |  可以支持在linux环境下启动  |  具体用法见[在线Serving部署](../doc/serving.md)  |
 |  static_ps_online_trainer.py  |  流式训练  |  可以在linux环境下从任意目录通过相对路径启动  |  具体用法见[流式训练](../doc/online_trainer.md)  |
  |  static_ps_offline_infer.py  |  基于参数服务器(ParameterServer)模式的分布式预测相关代码，目前仅支持使用静态图的方式预测  |  可以在linux环境下从任意目录通过相对路径启动  |  支持在任意目录下运行，以slot_dnn模型为示例，在PaddleRec根目录中运行的命令为：fleetrun --worker_num=1 --server_num=1 tools/static_ps_offline_infer.py -m models/rank/slot_dnn/config_offline_infer.yaml |
 |  to_static.py  |  动态图保存下来的模型模型转化为静态图进行inference  |  可以支持在linux环境下启动  |  具体用法见[python端预测库推理](../doc/inference.md)  |
 |  webserver.py  |  使用web方式启动serving服务端  |  可以支持在linux环境下启动  |  具体用法见[在线Serving部署](../doc/serving.md)  |
 |  onoff_diff  |  在线离线一致性检查  |  可以在linux环境下从任意目录通过相对路径启动  |  具体用法见[在线离线一致性检查](../doc/onoff_diff.md)  |
 
