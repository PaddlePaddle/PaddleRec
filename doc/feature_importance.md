# 特征重要性
本教程以[slot_dnn](../models/rank/slot_dnn/README.md)模型为例，介绍如何衡量已保存模型中特征的重要性。  
基本原理是：  
1、加载模型，在固定的测试集上进行预测，得到基线AUC值；  
2、shuffle数据中的某个slot，使该slot中的数据随机化，重新预测得到AUC值；  
3、比较基线AUC和slot_shuffle后的AUC，AUC下降幅度越大，特征越重要。  

## 配置
流式训练配置参见models/rank/slot_dnn/config_offline_infer.yaml，新增配置及作用如下：
|             名称              |     类型     |                           取值                            | 是否必须 |                               作用描述                               |
| :---------------------------: | :----------: | :-------------------------------------------------------: | :------: | :------------------------------------------------------------------: |
|             init_model_path              |    string    |                           任意                            |    是    |                            已保存的模型路径                            |
|             shots_shuffle_list              |    list    |                           任意                            |    是    |                需要衡量的特征列表                    |
|             candidate_size              |            int                  |                            任意                           |    是    |                需要slot_shuffle的样例个数                 |


## 使用方法
请在models/rank/slot_dnn目录下执行如下命令，启动特征重要性衡量脚本。  
```bash
fleetrun --server_num=1 --worker_num=1 ../../../tools/feature_importance.py -m config_offline_infer.yaml
```
日志中首先会对原始数据集进行预测，打印基准AUC；然后依次遍历shots_shuffle_list中的slot或者slot组合，对其进行slot_shuffle，预测后打印出相应的AUC，并计算AUC降幅，用户可通过降幅大小衡量特征的重要性。  
