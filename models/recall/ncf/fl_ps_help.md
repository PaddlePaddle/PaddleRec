# 1、功能介绍
基于 GEO-PS 实现的 FL-PS，支持 Coordinator：  
* 构造 worker 上异构样本数据
* 每一轮训练（Epoch）时，打印训练指标（loss、auc）
* 每一轮训练之后，用测试集数据推理

# 2、样本准备
* 在 PaddleRec/datasets/movielens_pinterest_NCF 目录中执行: sh run.sh，获取初步处理过的训练数据（big_train）和测试数据（test_data）
* 从 MovieLens 官网下载 ml-1m 数据集，获取 users.dat 文件（可自定义存储路径，但需要和 gen_heter_data.py 脚本中路径保持一致），后续用于构造异构数据集（按 zipcode 的首位数字划分）
* 在 PaddleRec/datasets/movielens_pinterest_NCF/fl_data 中新建目录 fl_test_data 和 fl_train_data，用于存放每个 client 上的训练数据集和测试数据集
* 在 PaddleRec/datasets/movielens_pinterest_NCF/fl_data 目录中执行: python gen_heter_data.py，生成 10 份训练数据
    * 总样本数 4970844（按 1:4 补充负样本）：0 - 518095，1 - 520165，2 - 373605，3 - 315550，4 - 483779，5 - 495635，6 - 402810，7 - 354590，8 - 262710，9 - 1243905
    * 样本数据每一行表示：物品 id，用户 id，标签
    
# 3、运行命令
1. 不带 coordinator 版本
* 在本文件所在的目录下执行：fleetrun --worker_num=10 --server_num=1 ../../../tools/static_fl_trainer.py -m config_fl.yaml
2. 带 coordinator 版本
* 在本文件所在的目录下执行：fleetrun --worker_num=10 --server_num=1 --coordinator_num=1 ../../../tools/static_fl_trainer.py -m config_fl.yaml
（可参考 fl_run.sh 文件）

# 4、二次开发
## 系统层面
1. 代码 repo
* Paddle: https://github.com/ziyoujiyi/Paddle/tree/fl_ps
* PaddleRec：https://github.com/ziyoujiyi/PaddleRec/tree/fl-rec
2. 编译、安装
```
1）去 https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/compile/linux-compile.html 找到 develop/Linux/源码编译/CPU/ 的开发镜像，在 docker 中开发
2）在 Paddle 根目录下，新建 build 目录
3）cd build
4）cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_DISTRIBUTE=ON -DWITH_PSCORE=ON -WITH_AVX=OFF -DWITH_TESTING=OFF -DWITH_FLPS=ON
5) make -j 
6）python -m pip install python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl --force-reinstall
```
3. 用户二次开发模块
* Paddle：
    * Paddle/python/paddle/distributed/ps/coordinator.py
    * 模型组网文件参考：PaddleRec/models/recall/ncf/net.py，用户如果新增组网文件，用前缀 "fl_" 标识
    * 数据集：如果 PaddleRec 中已经有的，直接在对应目录下新增 fl_test_data 和 fl_test_train 目录；如果 PaddleRec 中没有，用户在 PaddleRec/datasets 中新增
    * 用户自定义异构数据集构造，参考 gen_heter_data.py
    * 构造模型输入请参考：PaddleRec/models/recall/ncf/queuedataset_reader.py
4. 编码规范
* 风格检查：pip install pre-commit && 在 git 根目录下执行：pre-commit install
* 遵循已有风格


## 策略层面
1. 边缘任务调度策略
* 用户组网：DDPG
* 用户 python 端调用 _pull_dense 接口从 ps 端拉取 dense 参数，然后从 scope 里读
* 用户确定每轮训练之后 client 需要上传给 coordinator 的各个参数（字段）
2. 新的损失函数设计
* 直接使用不带 coordinator 版本的训练脚本
3. 知识蒸馏
* 用户训练 student 模型，打印 logits 结果，并上传到 coordinator，coordinator 端进行 teacher 模型训练
* coordinator 下发全局软目标
4. 模型压缩
