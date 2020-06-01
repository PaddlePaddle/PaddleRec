# MMoE

## 简介
多任务模型通过学习不同任务的联系和差异，可提高每个任务的学习效率和质量。多任务学习的的框架广泛采用shared-bottom的结构，不同任务间共用底部的隐层。这种结构本质上可以减少过拟合的风险，但是效果上可能受到任务差异和数据分布带来的影响。  论文[《Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts》]( https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture- )中提出了一个Multi-gate Mixture-of-Experts(MMOE)的多任务学习结构。MMOE模型刻画了任务相关性，基于共享表示来学习特定任务的函数，避免了明显增加参数的缺点。

## 快速开始
PaddleRec内置了demo小数据方便用户快速使用模型，训练命令如下

```shell
python -m paddlerec.run -m paddlerec.models.multitask.mmoe
```

## 模型效果

根据原论文，我们在开源数据集Census-income Data上验证模型效果

参数见config.yaml中的hyper_parameters部分，batch_size:32 epochs:400

两个任务的auc分别为：

1.income

max_mmoe_test_auc_income：0.94937 mean_mmoe_test_auc_income：0.94465

2.marital

max_mmoe_test_auc_marital：0.99419 mean_mmoe_test_auc_marital：0.99324
