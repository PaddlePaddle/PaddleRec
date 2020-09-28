# TDM-Demo建树及训练

## 建树所需环境
Requirements:
- python == 2.7
- paddlepaddle >= 1.7.2（建议1.7.2）
- paddle-rec （克隆github paddlerec，执行python setup.py install）
- sklearn
- anytree


## 建树流程

### 生成建树所需Embedding

- 生成Fake的emb

```shell
cd gen_tree
python -u emb_util.py
```

生成的emb维度是[13, 64]，含义是共有13个item，每个item的embedding维度是64，生成的item_emb位于`gen_tree/item_emb.txt`

格式为`emb_value_0(float) 空格 emb_value_1(float) ... emb_value_63(float) \t item_id `

在demo中，要求item的编号从0开始，范围 [0, item_nums-1]

真实场景可以通过各种hash映射满足该要求

### 对Item_embedding进行聚类建树

执行

```shell
cd gen_tree
# emd_path: item_emb的地址
# emb_size: item_emb的第二个维度，即每个item的emb的size（示例中为64）
# threads: 多线程建树配置的线程数
# n_clusters: 最终建树为几叉树，此处设置为2叉树
python gen_tree.py --emd_path item_emb.txt --emb_size 64 --output_dir ./output --threads 1 --n_clusters 2
```

生成的训练所需树结构文件位于`gen_tree/output`
```shell
.
├── id2item.json         # 树节点id到item id的映射表
├── layer_list.txt       # 树的每个层级都有哪些节点
├── travel_list.npy      # 每个item从根到叶子的遍历路径，按item顺序排序
├── travel_list.txt      # 上个文件的明文txt
├── tree_embedding.txt   # 所有节点按节点id排列组成的embedding
├── tree_emb.npy         # 上个文件的.npy版本
├── tree_info.npy        # 每个节点：是否对应item/父/层级/子节点，按节点顺序排列
├── tree_info.txt        # 上个文件的明文txt
└── tree.pkl             # 聚类得到的树结构
```

我们最终需要使用建树生成的以下四个文件，参与网络训练，参考`models/treebased/tdm/config.yaml`

1. layer_list.txt
2. travel_list.npy 
3. tree_info.npy
4. tree_emb.npy


### 执行训练

- 更改`config.yaml`中的配置

首先更改
```yaml
hyper_parameters:
    # ...
    tree:
        # 单机训练建议tree只load一次，保存为paddle tensor，之后从paddle模型热启
        # 分布式训练trainer需要独立load 
        # 预测时也改为从paddle模型加载
        load_tree_from_numpy: True # only once
        load_paddle_model: False # train & infer need, after load from npy, change it to True
        tree_layer_path: "{workspace}/tree/layer_list.txt"
        tree_travel_path: "{workspace}/tree/travel_list.npy"
        tree_info_path: "{workspace}/tree/tree_info.npy"
        tree_emb_path: "{workspace}/tree/tree_emb.npy"
```
将上述几个path改为建树得到的文件所在的地址

再更改
```yaml
hyper_parameters:
  max_layers: 4                          # 不含根节点，树的层数
  node_nums: 26                          # 树共有多少个节点，数量与tree_info文件的行数相等
  leaf_node_nums: 13                     # 树共有多少个叶子节点
  layer_node_num_list: [2, 4, 8, 10]     # 树的每层有多少个节点
  child_nums: 2                          # 每个节点最多有几个孩子结点（几叉树）
  neg_sampling_list: [1, 2, 3, 4]        # 在树的每层做多少负采样，训练自定义的参数
```

若并不知道对上面几个参数具体值，可以试运行一下，paddlerec读取建树生成的文件后，会将具体信息打印到屏幕上，如下所示：
```shell
...
File_list: ['models/treebased/tdm/data/train/demo_fake_input.txt']
2020-09-10 15:17:19,259 - INFO - Run TDM Trainer Startup Pass
2020-09-10 15:17:19,283 - INFO - load tree from numpy
2020-09-10 15:17:19,284 - INFO - TDM Tree leaf node nums: 13
2020-09-10 15:17:19,284 - INFO - TDM Tree max layer: 4
2020-09-10 15:17:19,284 - INFO - TDM Tree layer_node_num_list: [2, 4, 8, 10]
2020-09-10 15:17:19,285 - INFO - Begin Save Init model.
2020-09-10 15:17:19,394 - INFO - End Save Init model.
Running SingleRunner.
...
```
将其抄到配置中即可

- 训练

执行
```
cd /PaddleRec # PaddleRec 克隆的根目录
python -m paddlerec.run -m models/treebased/tdm/config.yaml
```
