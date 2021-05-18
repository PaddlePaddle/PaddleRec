# 使用昆仑XPU芯片加速NAML模型训练

## 准备Paddle昆仑XPU版训练环境
[昆仑XPU芯片运行飞桨](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/xpu_docs/index_cn.html)

## 数据准备

### 示例数据
参考 [数据准备](https://github.com/PaddlePaddle/PaddleRec/blob/master/models/rank/naml/README.md#数据准备)


### 全量数据
```shell
cd PaddleRec/datasets/MIND/data
bash run.sh
```

## 训练
```shell
# 设置训练使用的昆仑XPU芯片卡号
export FLAGS_selected_xpus=0
# 开启昆仑XPU芯片卷积计算加速(可不设置)
export XPU_CONV_AUTOTUNE=2

cd PaddleRec/models/rank/naml 
# 全量数据静态图训练
python3.7 -u ../../../tools/static_trainer.py -m config_bigdata_kunlun.yaml # 使用示例数据，请指定config_kunlun.yaml
```

## 评估
```shell
# 设置训练使用的昆仑XPU芯片卡号
export FLAGS_selected_xpus=0
# 开启昆仑XPU芯片卷积计算加速(可不设置)
export XPU_CONV_AUTOTUNE=2

cd PaddleRec/models/rank/naml 
# 全量数据静态图预测
python3.7 -u ../../../tools/static_infer.py -m config_bigdata_kunlun.yaml # 使用示例数据，请指定config_kunlun.yaml
```

## 模型效果
以下为全量数据训练2个epoch的结果:

| 模型 | 训练auc |batch_size | epoch_num| Time of each epoch| 
| :------| :------ | :------ | :------| :------ | 
| naml | 0.71 | 50 | 2 | 约7小时 | 


| 模型 | 预测auc |batch_size | Time of each epoch| 
| :------| :------ | :------ | :------ | 
| naml | 0.67 | 10 | 约2小时 | 
