import random
from pprint import pprint

from milvus import Milvus, DataType

_HOST = '172.17.0.1'
_PORT = '19530'
client = Milvus(_HOST, _PORT)

collection_name = 'demo_films'
if collection_name in client.list_collections():
    client.drop_collection(collection_name)

collection_param = {
    "fields": [
         #{"name": "id", "type": DataType.INT32},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 32}},
    ],
    "segment_row_limit": 16384,
    "auto_id": False
}

client.create_collection(collection_name, collection_param)
client.create_partition(collection_name, "Movie")

print("--------get collection info--------")
collection = client.get_collection_info(collection_name)
pprint(collection)
partitions = client.list_partitions(collection_name)
print("\n----------list partitions----------")
pprint(partitions)


import codecs
with codecs.open("movie_vectors.txt", "r",encoding='utf-8',errors='ignore') as f:
    lines = f.readlines()

ids = [int(line.split(":")[0]) for line in lines]
embeddings = []
for line in lines:
    line = line.strip().split(":")[1][1:-1]
    str_nums = line.split(",")
    emb = [float(x) for x in str_nums]
    embeddings.append(emb)
hybrid_entities = [
    #{"name": "id", "values": ids, "type": DataType.INT32},
    {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
]

ids = client.insert(collection_name, hybrid_entities, ids, partition_tag="Movie")
print("\n----------insert----------")
#print("Films are inserted and the ids are: {}".format(ids))
before_flush_counts = client.count_entities(collection_name)
client.flush([collection_name])
after_flush_counts = client.count_entities(collection_name)
print("\n----------flush----------")
print("There are {} films in collection `{}` before flush".format(before_flush_counts, collection_name))
print("There are {} films in collection `{}` after flush".format(after_flush_counts, collection_name))

films = client.get_entity_by_id(collection_name, ids=[1, 200])
print("\n----------get entity by id = 1, id = 200----------")
for film in films:
    if film is not None:
        print(" > id: {},  embedding: {}\n"
              .format(film.id, film.embedding))

query_embedding = [random.random() for _ in range(32)]
query_hybrid = {
    "bool": {
        "must": [
            {
                "vector": {
                    "embedding": {"topk": 100, "query": [query_embedding], "metric_type": "L2"}
                }
            }
        ]
    }
}

#
#     Now we've gotten the results, and known it's a 1 x 1 structure, how can we get ids, distances and fields?
#     It's very simple, for every `topk_film`, it has three properties: `id, distance and entity`.
#     All fields are stored in `entity`, so you can finally obtain these data as below:
#     And the result should be film with id = 3.
# ------
results = client.search(collection_name, query_hybrid, fields=["embedding"])
print("\n----------search----------")
for entities in results:
    for topk_film in entities:
        current_entity = topk_film.entity
        print("- id: {}".format(topk_film.id))
        print("- distance: {}".format(topk_film.distance))


