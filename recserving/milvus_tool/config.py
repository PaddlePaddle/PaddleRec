import os
from milvus import MetricType, IndexType

MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = 19530

collection_param = {
    'dimension': 128,
    'index_file_size': 2048,
    'metric_type': MetricType.L2
}

index_type = IndexType.IVF_FLAT
index_param = {'nlist': 1000}

top_k = 100
search_param = {'nprobe': 20}

