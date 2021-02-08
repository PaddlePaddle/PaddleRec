# PaddleRec 基于 Movielens 数据集的全流程示例

## 模型的详细教程可以查阅： [告别电影荒，手把手教你训练符合自己口味的私人电影推荐助手](https://aistudio.baidu.com/aistudio/projectdetail/1481839)

## 本地运行环境

PaddlePaddle>=2.0

python 2.7/3.5/3.6/3.7

os : linux 

本地运行流程与AiStudio流程基本一致，细节略有区别

## 数据处理
```shell
pip install py27hash
bash data_prepare.sh
```

### 模型训练
```shell
# 动态图训练recall模型
python -u ../../../tools/trainer.py -m recall/config.yaml
# 静态图训练recall模型
python -u ../../../tools/static_trainer.py -m recall/config.yaml
# 动态图训练rank模型
python -u ../../../tools/trainer.py -m rank/config.yaml
# 静态图训练rank模型
python -u ../../../tools/static_trainer.py -m rank/config.yaml
```

### 模型测试
```shell
# 动态图预测recall模型
python -u infer.py -m recall/config.yaml
# 静态图预测recall模型
python -u static_infer.py -m recall/config.yaml
# 动态图预测rank模型
python -u infer.py -m rank/config.yaml
# 静态图预测rank模型
python -u static_infer.py -m rank/config.yaml
```

### 测试结果解析
```shell
# recall模型的测试结果解析
python parse.py recall_offline recall_infer_result
# rank模型的测试结果解析
python parse.py rank_offline rank_infer_result
```
