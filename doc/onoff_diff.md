# 在线离线一致性检查
在线上服务搭建完成之后，为保证正确性，需要对同一批数据进行在线和离线预测，并对比结果，结果一致，才能启动线上服务。  
需要注意的是，在线/离线两个环境，预测的数据、加载的模型需保持一致。  

## 在线推理
参见在线推理模块(tools/inference)。  
需要开启debug模式，数据推理过程中打印每一层的输出tensor。  
如果在线推理是通过kv存储获取embedding向量，则还需要将数据中的所有feasign对应的embedding输出到文件中。  

## 离线预测
如果sparse embedding过大，离线预测无法单机进行，则需要从线上拿到预测数据所有feasign对应的embedding，并转换成离线能够加载的模型格式，这部分操作在get_xbox_model.py脚本中进行。  
离线预测使用tools/static_ps_offline_infer.py脚本，开启dump功能，打印每一层的输出tensor。  

## 一致性检查
得到在线/离线的预测结果之后，使用onoff_diff.py进行一致性检查。  

## 具体操作
本教程以[slot_dnn](../models/rank/slot_dnn/README.md)模型为例，介绍在线离线一致性检查的具体操作  
1. 启动前准备：  
    准备预测数据，放入在线环境相应位置及离线slot_dnn目录的infer_data/online下，同时在离线slot_dnn目录中建立infer_data/offline空文件夹  
    将离线训练好的模型cp到在线环境中（包括slot_dnn目录下的all_vars.txt文件及模型保存目录下的model和参数文件）  
2. 启动在线推理：生成cube.result及std.log文件  
3. 将上述两个文件cp至tools/onoff_diff/data文件夹下，std.log重命名为log.online  
4. 在tools/onoff_diff目录下运行sh run.sh，完成离线预测及在线离线一致性检查工作  
