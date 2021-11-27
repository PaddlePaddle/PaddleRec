=========================================================
PaddleRec 模型库
=========================================================
包含内容理解、匹配、召回、排序、 多任务、重排序等多个任务的完整推荐搜索算法库
以模型的分类为目录，模型名称为子目录，介绍各个模型
这些模型的简要介绍如下：


内容理解
--------------------------
我们提供了常见的内容理解任务中使用的模型算法的PaddleRec实现, 包括动态图和静态图的单机训练&预测效果指标。实现的内容理解模型包括 [Tagspace](tagspace)、[文本分类](textcnn)等。
模型算法库在持续添加中，欢迎关注。

.. toctree::
   :maxdepth: 1
   :caption: 内容理解模型
   :name: contentunderstanding

   contentunderstanding/tagspace.md
   contentunderstanding/textcnn.md

匹配
--------------------------
我们提供了常见的匹配任务中使用的模型算法的PaddleRec实现, 包括动态图和静态图的单机训练&预测效果指标。实现的模型包括 [DSSM](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/match/dssm)、[MultiView-Simnet](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/match/multiview-simnet)、[match-pyramid](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/match/match-pyramid)。  
模型算法库在持续添加中，欢迎关注。

.. toctree::
   :maxdepth: 1
   :caption: 匹配模型
   :name: match

   match/dssm.md
   match/match-pyramid.md
   match/multi-sminet.md


召回
--------------------------


排序
--------------------------


重排序
--------------------------

