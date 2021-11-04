# README

Milvus 是一款开源的特征向量相似度搜索引擎。本工具是基于 Milvus1.1.1 实现的提供向量存储与召回的服务。你可以将本工具用在推荐系统中的召回这一过程。

Milvus 教程请参考官网：https://milvus.io/cn/

Milvus 源码详情参考：https://github.com/milvus-io/milvus

## 目录结构

以下是本工具的简要目录结构及说明：

```text
├── readme.md #介绍文档
├── config.py #参数配置
├── milvus_insert.py  # 向量插入脚本
├── milvus_recall.py  # 向量召回脚本
├── milvus_helper.py  # Milvus 常用操作
```

## 环境要求

**操作系统**

CentOS: 7.5 或以上

Ubuntu LTS： 18.04 或以上

**硬件**

cpu: Intel CPU Sandy Bridge 或以上

> 要求 CPU 支持以下至少一个指令集： SSE42, AVX, AVX2, AVX512

内存： 8GB 或以上 （取决于具体向量数据规模）

**软件**

Python 版本： 3.6 及以上

Docker: 19.03 或以上

Milvus 1.0.0



## 安装启动 Milvus

这里将安装 [Milvus1.1.1 的 CPU 版本](https://milvus.io/cn/docs/v1.1.1/milvus_docker-cpu.md)，也可以选择安装 GPU 版本的 Milvus，安装方式请参考： [Milvus1.1.1 GPU 安装](https://milvus.io/cn/docs/v1.1.1/milvus_docker-gpu.md)。

**拉取 CPU 版本的 Milvus 镜像：**

```shell
$ sudo docker pull milvusdb/milvus:1.1.1-cpu-d061621-330cc6
```

**下载配置文件**

```shell
$ mkdir -p /home/$USER/milvus/conf
$ cd /home/$USER/milvus/conf
$ wget https://raw.githubusercontent.com/milvus-io/milvus/v1.1.1/core/conf/demo/server_config.yaml
```

> Milvus 相关的配置可以通过该配置文件指定。

**启动 Milvus Docker 容器**

```shell
$ sudo docker run -d --name milvus_cpu_1.1.1 \
-p 19530:19530 \
-p 19121:19121 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
-v /home/$USER/milvus/wal:/var/lib/milvus/wal \
milvusdb/milvus:1.1.1-cpu-d061621-330cc6
```

**确认 Milvus 运行状态**

```shell
$ sudo docker logs milvus_cpu_1.1.1
```

> 用以上命令查看 Milvus 服务是否正常启动。

**安装 Milvus Python SDK**

```shell
$ pip install pymilvus==1.1.2
```



## 使用说明

本工具中的脚本提供向量插入和向量召回两个功能。在使用该工具的脚本前，需要先根据环境修改该工具中的配置文件 `config.py`：

| Parameters       | Description                                                  | Reference value                                              |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MILVUS_HOST      | Milvus 服务所在的机器 IP                                     | 127.0.0.1                                                    |
| MILVUS_PORT      | 提供 Milvus 服务的端口                                       | 19530                                                        |
| collection_param | 在 Milvus 中建立的集合参数。<br />`dimension` 表示向量维度<br />`index_file_size` 表示在 Milvus 中存储的数据文件大小<br />`metric_type` 表示计算向量相似度的方式 | collection_param = {<br />      'dimension': 128,<br />      'index_file_size': 2048,<br />      'metric_type': MetricType.L2} |
| index_type       | 指定给 Milvus 集合建立的索引类型                             | IndexType.IVF_FLAT                                           |
| index_param      | 建立索引的参数，不同索引所需要的参数不同                     | {'nlist': 1000}                                              |
| top_k            | 查询时，召回的向量数。                                       | 100                                                          |
| search_param     | 在 Milvus 中查询时的参数，该参数会影像查询性能和召回率       | {'nprobe': 20}                                               |

### 向量导入

`milvus_insert.py` 脚本提供向量导入功能，在使用该脚本前，需要在config.py 修改对应参数。调用方式如下：

```python
from milvus_tool.milvus_insert import VecToMilvus

client = VecToMilvus()
status, ids = client.insert(collection_name=collection_name, vectors=embeddings, ids=ids, partition_tag=partition_name)
```

> 调用 insert 方法时需要传入的参数：
>
> **collection_name**: 将向量插入 Milvus 中的集合的名称。该脚本在导入数据前，会检查库中是否存在该集合，不存在的话会按照 `config.py` 中设置的集合参数建立一个新的集合。
>
> **vectors**: 插入 Milvus 集合中的向量。这里要求的是向量格式是二维列表的形式，示例：[[2.1, 3.2, 10.3, 5.5], [3.3, 4.2, 6.5, 6.3]] ，这里表示插入两条维度为四的向量。
>
> **ids**: 和向量一一对应的 ID，这里要求的 ids 是一维列表的形式，示例：[1,2]，这里表示上述两条向量对应的 ID 分别是 1 和 2. 这里的 ids 也可以为空，不传入参数，此时插入的向量将由 Milvus 自动分配 ID。
>
> **partition_tag**: 指定向量要插入的分区名称，Milvus 中可以通过标签将一集合分割为若干个分区 。该参数可以为空，为空时向量直接插入集合中。
>
> 在像 Milvus 指定的集合 collection 或者分区 partition 中插入参数时，如果 Milvus 中不存在该集合或者分区，该脚本会自动建立对应的集合或者分区。

**返回结果**：向量导入后将返回 `status` 和 `ids` 两个参数。status 返回的是插入的状态，插入成功或者失败。ids 返回的是插入向量对应的 ID，是一个一维列表。

具体使用可参考项目 movie_recommender/to_milvus.py

### 向量召回

`milvus_recall.py` 提供向量召回功能，在使用该脚本前，需要在config.py 修改对应参数，调用方式如下：

```python
from milvus_tool.milvus_recall import RecallByMilvus
milvus_client = RecallByMilvus()
status, results = self.milvus_client.search(collection_name=collection_name, vectors = query_records, partition_name=partition_name)
```

> **collection_name**：指定要查询的集合名称。
>
> **vectors**：指定要查询的向量。该向量格式和插入时的向量格式一样，是一个二维列表。
>
> **partition_tag**：指定查询的分区标签。该参数可以为空，不指定时在 collection 的全局范围内查找。

**返回结果**：查询后将返回 `status` 和 `results` 两个结果。`status` 返回的是查询的状态，查询成功或者失败。`results` 返回的是查询结果，返回结果的格式示例：

```
以下查询两条向量，top_k =3 是的结果示例
[
 [ (id:000, distance:0.0),
   (id:020, distance:0.17),
   (id:111, distance:0.60) ]
 [ (id:100, distance:0.0),
   (id:032, distance:0.69),
   (id:244, distance:1.051) ]
 ]
```

具体使用可参考项目 movie_recommender/recall.py

### Milvus 基本操作

`milvus_helper.py` 脚本中提供了以下几个 Milvus 常用操作：

- 在 Milvus 中建立 collection
- 查看 Milvus 中是否存在指定 collection
- 查看指定 collection 中导入的向量数
- 查看 Milvus 中所有的 collection
- 删除指定 collection

调用方式如下：

```python
from milvus_tool.milvus_helper import MilvusHelper
client = MilvusHelper()
collection_name = 'test'
```

- 查看 Milvus 中是否存在某 collection

```python
print(client.has_collection(collection_name))
```

- 在 Milvus 中建立 collection，建立 collection 的参数可修改 `config.py` 中的 `collection_param`

```python
client.creat_collection(collection_name)
```

- 查看指定 collection 中的向量数

```python
print(client.count(collection_name))
```

- 查看 Milvus 中所有的 collection

```python
print(client.list_collection())
```

- 删除 Milvus 中的指定 collection

```python
client.delete_collection(collection_name)
```
