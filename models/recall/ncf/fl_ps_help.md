# 功能介绍
1. 单机版本
2. 单机模拟分布式：async-ps
    * worker 数 1 个
    * worker 数 多个 - 样本复制
    * worker 数 多个 - 样本切分
    * 打印训练指标（loss、auc）
    * 推理
3. 单机模拟分布式：geo-ps
    * worker 数 1 个
    * worker 数 多个
    * 构造 worker 上异构样本数据
    * heter-aware ps
    * 打印训练指标（loss、auc）
    * 推理
4. pdc

# 样本处理
* 在 PaddleRec/datasets/movielens_pinterest_NCF/fl_data 目录中执行：python gen_heter_data.py
    * 总样本数 4970844（按 1:4 补充负样本）：0 - 518095，1 - 520165，2 - 373605，3 - 315550，4 - 483779，5 - 495635，6 - 402810，7 - 354590，8 - 262710，9 - 1243905
    
# 运行命令
* 在当前目录下执行：fleetrun --worker_num=1 --server_num=1 ../../../tools/static_fl_trainer.py -m config_fl.yaml
