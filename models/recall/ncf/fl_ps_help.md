# 功能介绍
1. 单机版本
2. 单机模拟分布式：async-ps
    * worker 数 1 个
    * worker 数 多个 - 样本复制
    * worker 数 多个 - 样本切分
    * 打印训练指标
3. 单机模拟分布式：geo-ps
    * worker 数 1 个
    * worker 数 多个
    * 构造 worker 上异构样本数据
    * heter-aware ps
    * 打印训练指标
4. pdc

# 运行命令
* 在当前目录下执行：fleetrun --worker_num=1 --server_num=1 ../../../tools/static_fl_trainer.py -m config_fl.yaml
