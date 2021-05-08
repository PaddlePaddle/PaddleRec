# Paddle TDM系列模型解决方案

本示例代码提供了基于PaddlePaddle实现的树模型推荐搜索算法，包括[TDM](https://arxiv.org/pdf/1801.02294.pdf)，[JTM](https://arxiv.org/pdf/1902.07565.pdf)。树模型是为大规模推荐系统设计的、能承载任意先进模型来高效检索用户兴趣的推荐算法解决方案。该方案基于树结构，提出了一套对用户兴趣度量进行层次化建模与检索的方法论，使得系统能直接利高级深度学习模型在全库范围内检索用户兴趣。其基本原理是使用树结构对全库item进行索引，然后训练深度模型以支持树上的逐层检索，从而将大规模推荐中全库检索的复杂度由O(n)（n为所有item的量级）下降至O(log n)。


## 快速开始

基于demo数据集，快速上手TDM系列模型，为您后续设计适合特定使用场景的模型做准备。

假定您PaddleRec所在目录为${PaddleRec_Home}。

- Step1: 进入tree-based模型库文件夹下，完成demo数据集的切分、建树等准备工作。

```shell
cd ${PaddleRec_Home}/models/treebased/
./data_prepare.sh demo
```
demo数据集预处理一键命令为 `./data_prepare.sh demo` 。若对具体的数据处理、建树细节感兴趣，请查看    `data_prepare.sh` 脚本。这一步完成后，您会在 `${PaddleRec_Home}/models/treebased/` 目录下得到一个名为 `demo_data`的目录，该目录结构如下：

```
├── treebased
├── demo_data
|   ├── samples                      JTM Tree-Learning算法所需，
|   |   ├── samples_{item_id}.json   记录了所有和item_id相关的训练集样本。
|   ├── train_data                   训练集目录
|   ├── test_data                    测试集目录
|   ├── ItemCate.txt                 记录所有item的类别信息，用于初始化建树。
|   ├── Stat.txt                     记录所有item在训练集中出现的频次信息，用于采样。
|   ├── tree.pb                      预处理后，生成的初始化树文件
```

- Step2: 训练。config.yaml中配置了模型训练所有的超参，运行方式同PaddleRec其他模型静态图运行方式。当前树模型暂不支持动态图运行模式。

```shell
python -u ../../../tools/static_trainer.py -m config.yaml 
```

- Step3: 预测，命令如下所示。其中第一个参数为训练config.yaml位置，第二个参数为预测模型地址。

```
python infer.py config.yaml ./output_model_tdm_demo/0/
```

- Step4: 提取Item（叶子节点）的Embedding，用于重新建树，开始下一轮训练。命令如下所示，其中第一个参数为训练config.yaml位置，第二个参数模型地址，第三个参数为输出文件名称。

```
python get_leaf_embedding.py config.yaml  ./output_model_tdm_demo/0/ epoch_0_item_embedding.txt
```

- Step5: 基于Step4得到的Item的Embedding，重新建树。命令如下所示。

```
cd ../builder && python tree_index_builder.py --mode by_kmeans --input epoch_0_item_embedding.txt --output new_tree.pb
```

- Step6: 修改config.yaml中tree文件的路径为最新tree.pb，返回Step2，开始新一轮的训练。
