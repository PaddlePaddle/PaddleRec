# PaddleRec 贡献代码

本章介绍如何开发并提交一个模型，以动态图为例

## 提交方式

PaddleRec套件的准则基本和主Paddle的一致，请参考[官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/10_contribution/local_dev_guide_cn.html)

请使用Pre-commit钩子，否则CI代码检测不会通过！

请使用Pre-commit钩子，否则CI代码检测不会通过！

请使用Pre-commit钩子，否则CI代码检测不会通过！

## 代码风格

Python代码风格，请参考主流的[开源风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)

## 模型示例

一个完整的模型示例请参考[MMoE](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe)

## 训练/预估执行器

通用的执行器在tools目录下，分别对应trainer.py/infer.py

Tips1: 有一些特殊的处理逻辑或者评估方法，可以在模型目录下面单独添加执行器，如models/recall/word2vec/infer.py

Tips2: 如果是自己的执行器，日志规范请参考通用执行器。

## 模型目录

具体的模型目录取决于模型的应用方向，比如rank/recall/multitask等，需要在模型目录下创建__init__.py文件

Tips1: 如果不确定在哪个目录的可以先按照Rank目录提交PR，后续Review过程中再修改

Tips2: 如有条件可同步修改首页模型目录指引

## 文档Readme

具体请参考[MMoE](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/multitask/mmoe/README.md)

需要至少包含模型介绍，运行环境，数据处理，快速开始，模型组网，复现结果

Tips1: 如果有条件请同步添加英文文档Readme_en.md

Tips2: 文档很重要，文档很重要，文档很重要！重要的事情说三遍

Tips3: 可运行命令部分需要使用```bash格式，可参考MMoE的README命令部分

## 数据

样例数据统一以sample_data命名，在模型目录下面，只支持小样本数据

全量数据请放到根目录下的dataset目录，详细说明参考[这里](https://github.com/PaddlePaddle/PaddleRec/tree/master/datasets)

Tips1: sample_data需要足够小，否则pre-commit会报文件太大的错误

Tips2: 全量数据如需要存放在云端，可联系我们

## 数据读取Reader

PaddleRec对主流的数据集提供了相关的Reader脚本，可参考相关模型下的xxx_reader.py, 复制到模型目录下

如需开发自定义Reader，请参考[Reader](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/custom_reader.md)

## 模型组网net

考虑到动静统一的需求，我们将模型组网的逻辑拆分成两部分 net.py 和 dygraph_model.py(动态图)

Tips1: 模型文件名必须是net.py和dygraph_model.py

### net.py

采用Class方式，嵌套定义模型，Class类必须继承paddle.nn.Layer

__init__方法定义参数，forward方法定义前向计算逻辑

Tips1: 对于数组定义而不是self指向的参数，需要显示通过add_sublayer()方法关联到此模型，否则模型class找不到这个参数

Tips2: 由于net.py中是动态图和静态图通用组网，所以要求使用的API必须是2.0以后的API, 同时不能出现fluid 旧的字样

### dygraph_model.py

dygraph_model包含了数据feed逻辑，loss计算逻辑，优化器定义逻辑和指标定义逻辑，详细介绍参考[如何添加自定义模型](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/model_develop.md)


## 配置文件yaml

为了更方便的快速运行和实现论文精度，我们要求需要提供两个yaml文件，分别是config.yaml和config_bigdata.yaml

yaml的详细配置说明请参考[config.yaml配置说明](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/yaml.md)

Tips1: config_bigdata中的数据路径请配置dataset的相对路径，方便一键执行

Tips2: config_bigdata中的模型保存路径请设置为output_model_模型名_all的形式，区别于config中模型保存路径的output_model_模型名

## 将模型信息添加到主页
为了方便用户在主页了解到模型分布情况，需要在主页(包括中文和英文)的支持模型列表添加该模型信息

英文首页参考[Support Model List](https://github.com/PaddlePaddle/PaddleRec#support-model-list)

中文首页参考[中文首页支持模型列表](https://github.com/PaddlePaddle/PaddleRec/blob/master/README_CN.md#%E6%94%AF%E6%8C%81%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8)

注意：中英文两部分的模型信息顺序须保持一致

## 将模型添加到PaperWithCode
进入[PaperWithCode](https://paperswithcode.com/)

若无账号须要先申请

搜索模型对应论文名称，进入该页面即可添加模型

## 外部开发者列表
如果模型的贡献来自外部开发者，需将该作者信息添加到[外部开发者列表](../contributor.md)

## 添加TIPC
为了方便检测模型的精度变化及运行情况，需要模型添加tipc的配置并支持tipc的脚本运行

添加新的模型需要支持lite_train_lite_infer，lite_train_whole_infer，whole_infer，whole_train_whole_infer四种模式。详细运行方式可以参考[基础训练预测功能测试](https://github.com/PaddlePaddle/PaddleRec/blob/master/test_tipc/doc/test_train_inference_python.md)

tips1：更改test_tipc/prepare.sh脚本，在最后添加自己的模型在四种模式时需要下载的数据集，以及用于全量数据预测时的模型下载链接

tips2：在test_tipc/configs目录下为自己的模型建立子目录，目录中需要有train_infer_python.txt文件以运行测试脚本、以及测试脚本运行时需要用到的独立脚本

PS:模型若需要存放于云端可以联系我们

## 添加AIStudio在线运行示列

## FAQ
