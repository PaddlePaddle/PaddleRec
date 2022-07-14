# TIPC Linux端Benchmark测试文档

该文档为Benchmark测试说明，Benchmark预测功能测试的主程序为`benchmark_train.sh`，用于验证监控模型训练的性能。

# 1. 测试流程
## 1.1 准备数据和环境安装
运行`test_tipc/prepare.sh`，完成训练数据准备和安装环境流程。

```shell
# 运行格式：bash test_tipc/prepare.sh  train_benchmark.txt  mode
bash test_tipc/prepare.sh test_tipc/configs/dnn/train_infer_python.txt benchmark_train
```

## 1.2 功能测试
执行`test_tipc/benchmark_train.sh`，完成模型训练和日志解析

```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_benchmark.txt mode
bash test_tipc/benchmark_train.sh test_tipc/configs/dnn/train_infer_python.txt benchmark_train
```

`test_tipc/benchmark_train.sh`支持根据传入的第三个参数实现只运行某一个训练配置，如下：
```shell
# 运行格式：bash test_tipc/benchmark_train.sh train_benchmark.txt mode
bash test_tipc/benchmark_train.sh test_tipc/configs/dnn/train_infer_python.txt benchmark_train null_bs8_null_null_N1C8
```
dynamic_bs8_fp32_DP_N1C1为test_tipc/benchmark_train.sh传入的参数，格式如下：
`${modeltype}_${batch_size}_${fp_item}_${run_mode}_${device_num}`
包含的信息有：模型类型、batchsize大小、训练精度如fp32,fp16等、分布式运行模式以及分布式训练使用的机器信息如单机单卡（N1C1）。


## 2. 日志输出
benchmark训练得到训练日志后，会自动保存训练日志并解析得到ips等信息, 在benchmark测试时，会自动调用{benchmark_root}/scrips/analysis.py

BENCHMARK_ROOT 通过设置环境变量的方式来设置，比如：
```
export BENCHMARK_ROOT=/paddle/PaddleRec/test_tipc
benchmark_train.sh在运行时会自动调用/paddle/PaddleRec/test_tipc/scripts/analysis.py
```
运行后将保存模型的训练日志和解析日志，使用 `test_tipc/configs/dnn/train_benchmark.txt` 参数文件的训练日志解析结果是：

```
{"model_branch": "gpups", "model_commit": "2ccd243761b39dffe037cef5160dda722f121311", "model_name": "dnn_bs2048_3_MultiP_DP", "batch_size": 2048, "fp_item": "3", "run_mode": "DP", "convergence_value": 0, "convergence_key": "loss:", "ips": 0, "speed_unit": "", "device_num": "N1C4", "model_run_time": "0", "frame_commit": "360b8383250774108a6561e7071d60189b0d0964", "frame_version": "0.0.0"}
```

训练日志和日志解析结果保存在benchmark_log目录下，文件组织格式如下：
```
train_log/
├── index
│   ├── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C1_speed
│   └── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C4_speed
├── profiling_log
│   └── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C1_profiling
└── train_log
    ├── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C1_log
    └── PaddleOCR_det_mv3_db_v2_0_bs8_fp32_SingleP_DP_N1C4_log
```
