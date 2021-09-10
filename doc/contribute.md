# PaddleRec 贡献代码

本章介绍如何开发并提交一个模型，以动态图为例

## 模型组成

一个完整的模型示例请参考[MMoE](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe)

### 训练/预估执行器

通用的执行器在tools目录下，分别对应trainer.py/infer.py

Tips1: 有一些特殊的处理逻辑或者评估方法，可以在模型目录下面单独添加执行器，如models/recall/word2vec/infer.py

Tips2: 如果是自己的执行器，日志规范请参考通用执行器。

### 模型目录

具体的模型目录取决于模型的应用方向，比如rank/recall/multitask等，需要在模型目录下创建__init__.py文件

Tips1: 如果不确定在哪个目录的可以先按照Rank目录提交PR，后续Review过程中再修改

Tips2: 如有条件可同步修改首页模型目录指引

### 文档Readme

具体请参考[MMoE](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe/README.md)

需要至少包含模型介绍，运行环境，数据处理，快速开始，模型组网，复现结果

Tips1: 如果有条件请同步添加英文文档Readme_en.md

Tips2: 文档很重要，文档很重要，文档很重要！重要的事情说三遍

### 数据
