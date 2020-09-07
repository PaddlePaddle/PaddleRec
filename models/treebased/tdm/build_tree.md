

wget https://paddlerec.bj.bcebos.com/utils/tree_build_utils.tar.gz --no-check-certificate

# input_path: embedding的路径
# emb_shape: embedding中key-value，value的维度
# emb格式要求： embedding_id(int64),embedding(float),embedding(float),......,embedding(float)
# cluster_threads: 建树聚类所用线程
python_172_anytree/bin/python -u main.py --input_path=./gen_emb/item_emb.txt --output_path=./ --emb_shape=24 --cluster_threads=4

建树流程是：1、读取emb -> 2、kmeans聚类 -> 3、聚类结果整理为树 -> 4、基于树结构得到模型所需的4个文件
    1    Layer_list：记录了每一层都有哪些节点。训练用
    2    Travel_list：记录每个叶子节点的Travel路径。训练用
    3    Tree_Info：记录了每个节点的信息，主要为：是否是item/item_id，所在层级，父节点，子节点。检索用
    4    Tree_Embedding：记录所有节点的Embedding。训练及检索用

注意一下训练数据输入的item是建树之前用的item id，还是基于树的node id，还是基于叶子的leaf id，在tdm_reader.py中，可以加载字典，做映射。
用厂内版建树得到的输出文件夹里，有名为id2nodeid.txt的映射文件，格式是『hash值』+ 『树节点ID』+『叶子节点ID（表示第几个叶子节点，tdm_sampler op 所需的输入）』
在另一个id2bidword.txt中，也有映射关系，格式是『hash值』+『原始item ID』，这个文件中仅存储了叶子节点的信息。
