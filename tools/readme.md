# PaddleRec小工具

## 工具简介
 |           程序名称            |                       实现的功能                    |                 支持的环境                 |            使用命令         |
 | :---------------------------: | :-----------------------------------------------: | :-------------------------------: |:-------------------------------: |
 |  trainer.py  |  trainer.py是使用动态图进行训练的相关代码，主要实现了动态图通用的训练流程  |  可以支持在windows/linux/macos环境下从任意目录通过相对路径启动。支持使用cpu/gpu运行  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：python -u tools/trainer.py -m models/rank/dnn/config.yaml  |
 |  infer.py  |  infer.py是使用动态图进行预测的相关代码，主要实现了动态图通用的预测流程  |  可以支持在windows/linux/macos环境下从任意目录通过相对路径启动。支持使用cpu/gpu运行  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：python -u tools/infer.py -m models/rank/dnn/config.yaml  |
 |  static_train.py  |  static_train.py是使用静态图进行训练的相关代码，主要实现了静态图通用的训练流程  |  可以支持在windows/linux/macos环境下从任意目录通过相对路径启动。支持使用cpu/gpu运行  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：python -u tools/static_trainer.py -m models/rank/dnn/config.yaml  |
 |  static_infer.py  |  static_infer.py是使用静态图进行预测的相关代码，主要实现了动态图通用的预测流程  |  可以支持在windows/linux/macos环境下从任意目录通过相对路径启动。支持使用cpu/gpu运行  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：python -u tools/static_infer.py -m models/rank/dnn/config.yaml  |
 |  static_ps_trainer.py  | static_ps_trainer.py是基于参数服务器模式(ParameterServer)的分布式训练相关代码，目前仅支持使用静态图的方式训练  |  可以支持在linux环境下从任意目录通过相对路径启动。  |  支持在任意目录下运行，以dnn模型为示例，在PaddleRec根目录中运行的命令为：fleetrun --worker_num=1 --server_num=1 tools/static_ps_trainer.py -m models/rank/dnn/config.yaml  |
