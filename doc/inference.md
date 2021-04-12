# Paddle Inference的使用方法
paddlerec目前提供在静态图训练时使用save_inference_model接口保存模型，以及将保存的模型使用Inference预测库进行服务端部署的功能。本教程将以wide_deep模型为例，说明如何使用这两项功能。  

## 使用save_inference_model接口保存模型
在服务器端使用python部署需要先使用save_inference_model接口保存模型。  
1. 首先需要在模型的yaml配置中，加入use_inference参数，并把值设置成True。use_inference决定是否使用save_inference_model接口保存模型，默认为否。若使用save_inference_model接口保存模型，保存下来的模型支持使用Paddle Inference的方法预测，但不支持直接使用paddlerec原生的的预测方法加载模型。  
2. 确定需要的输入和输出的预测模型变量，将其变量名以字符串的形式填入save_inference_feed_varnames和save_inference_fetch_varnames列表中。  
以wide_deep模型为例，可以在其config.yaml文件中观察到如下结构。训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。feed参数的名字中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示，```<integer feature>```代表数值特征（连续特征dense_input），共有13个连续特征，```<categorical feature>```代表分类特征（离散特征C1~C26），共有26个离散特征。fetch参数输出的是auc，具体意义为static_model.py里def net（）函数中将auc使用cast转换为float32类型语句中的cast算子。  
```yaml
runner:
  # 通用配置不再赘述
  ...
  # use inference save model
  use_inference: True  # 静态图训练时保存为inference model
  save_inference_feed_varnames: ["label","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","dense_input"] # inference model 的feed参数的名字
  save_inference_fetch_varnames: ["cast_0.tmp_0"] # inference model 的fetch参数的名字
```
3. 启动静态图训练
```bash
# 进入模型目录
# cd models/rank/wide_deep # 在任意目录均可运行
# 静态图训练
python -u ../../../tools/static_trainer.py -m config.yaml # 全量数据运行config_bigdata.yaml 
```

## 将保存的模型使用Inference预测库进行服务端部署
paddlerec提供tools/paddle_infer.py脚本，供您方便的使用inference预测库高效的对模型进行预测。  

1. 启动paddle_infer.py脚本的参数：

|        名称         |    类型    |             取值             | 是否必须 |                               作用描述                               |
| :-----------------: | :-------: | :--------------------------: | :-----: | :------------------------------------------------------------------: |
|       --model_file        |    string    |       任意路径         |    是    |                            模型文件路径（当需要从磁盘加载 Combined 模型时使用）                           |
|       --params_file        |    string    |       任意路径         |    是    |                            参数文件路径 （当需要从磁盘加载 Combined 模型时使用）                           |
|       --model_dir        |    string    |       任意路径         |    是    |                            模型文件夹路径 （当需要从磁盘加载非 Combined 模型时使用）                           |
|       --use_gpu        |    bool    |       True/False         |    是    |                            是否使用gpu                            |
|       --data_dir        |    string    |       任意路径         |    是    |                            测试数据目录                            |
|       --reader_file        |    string    |       任意路径         |    是    |                          测试时用的Reader()所在python文件地址                            |
|       --batchsize        |    int    |       >= 1         |    是    |                            批训练样本数量                            |
|       --model_name        |    str    |       任意名字         |    否    |                            输出模型名字                            |

2. 以wide_deep模型的demo数据为例，启动预测：
```bash
# 进入模型目录
# cd models/rank/wide_deep # 在任意目录均可运行
python -u ../../../tools/paddle_infer.py --model_file=output_model_wide_deep/2/rec_inference.pdmodel --params_file=output_model_wide_deep/2/rec_inference.pdiparams --use_gpu=False --data_dir=data/sample_data/train --reader_file=criteo_reader.py --batchsize=5
```
