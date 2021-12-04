=========================================================
PaddleRec 模型库
=========================================================
包含内容理解、匹配、召回、排序、 多任务、重排序等多个任务的完整推荐搜索算法库

以模型的分类为目录，模型名称为子目录，介绍各个模型

这些模型的简要介绍如下：


内容理解
--------------------------
我们提供了常见的内容理解任务中使用的模型算法的PaddleRec实现, 包括动态图和静态图的单机训练&预测效果指标。实现的内容理解模型包括 Tagspace、文本分类textcnn等。

模型算法库在持续添加中，欢迎关注。

.. toctree::
   :maxdepth: 1

   contentunderstanding/tagspace.md
   contentunderstanding/textcnn.md

匹配
--------------------------
我们提供了常见的匹配任务中使用的模型算法的PaddleRec实现, 包括动态图和静态图的单机训练&预测效果指标。实现的模型包括 DSSM、MultiView-Simnet、match-pyramid。  

模型算法库在持续添加中，欢迎关注。

.. toctree::
   :maxdepth: 1

   match/dssm.md
   match/match-pyramid.md
   match/multiview-simnet.md


召回
--------------------------
我们提供了常见的召回任务中使用的模型算法的PaddleRec实现, 单机训练&预测效果指标以及分布式训练&预测性能指标等。实现的召回模型包括 gru4rec、deepwalk、mind、ncf、word2vec等  

模型算法库在持续添加中，欢迎关注。

.. toctree::
   :maxdepth: 1

   recall/gru4rec.md
   recall/deepwalk.md
   recall/mind.md
   recall/ncf.md
   recall/word2vec.md


排序
--------------------------
我们提供了常见的排序任务中使用的模型算法的PaddleRec实现, 包括动态图和静态图的单机训练&预测效果指标。实现的排序模型包括 bst、dcn、deepfefm、deepfm、dien、difm、din、dlrm、dmr、dnn、ffm、fm、gatenet、logistic_regression、naml、wide_deep、xdeepfm等

模型算法库在持续添加中，欢迎关注。

.. toctree::
   :maxdepth: 1

   rank/bst.md
   rank/dcn.md
   rank/deepfefm.md
   rank/deepfm.md
   rank/dien.md
   rank/difm.md
   rank/din.md
   rank/dlrm.md
   rank/dmr.md
   rank/dnn.md
   rank/ffm.md
   rank/fm.md
   rank/gatenet.md
   rank/logistic_regression.md
   rank/naml.md
   rank/wide_deep.md
   rank/xdeepfm.md


重排序
--------------------------

