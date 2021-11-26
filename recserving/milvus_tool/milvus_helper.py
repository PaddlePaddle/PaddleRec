# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from milvus import Milvus, IndexType
from config import MILVUS_HOST, MILVUS_PORT, collection_param
import sys


class MilvusHelper():
    def __init__(self):
        try:
            self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
            print("Successfully connect to Milvus with IP:{} and PORT:{}".
                  format(MILVUS_HOST, MILVUS_PORT))
        except Exception as e:
            print("Failed to connect Milvus: {}".format(e))
            sys.exit(1)

    # Return if Milvus has the collection
    def has_collection(self, collection_name):
        try:
            status, ok = self.client.has_collection(collection_name)
            return ok
        except Exception as e:
            print("Failed to load data to Milvus: {}".format(e))
            sys.exit(1)

    # Create a collection in Milvus
    def creat_collection(self, collection_name):
        try:
            collection_param['collection_name'] = collection_name
            status = self.client.create_collection(collection_param)
            print(status)
        except Exception as e:
            print("Milvus create collection error:", e)

    # Delete Milvus collection
    def delete_collection(self, collection_name):
        try:
            if self.has_collection(collection_name):
                status = self.client.drop_collection(
                    collection_name=collection_name)
                print(status)
            else:
                print("collection {} doesn't exist".format(collection_name))
        except Exception as e:
            print("Failed to drop collection: {}".format(e))

    # Get the number of vectors in a collection
    def count(self, collection_name):
        try:
            if self.has_collection(collection_name):
                status, num = self.client.count_entities(
                    collection_name=collection_name)
                return num
            else:
                print("collection {} doesn't exist".format(collection_name))
                return 0
        except Exception as e:
            print("Failed to count vectors in Milvus: {}".format(e))
            sys.exit(1)

    # Show all collections in Milvus server
    def list_collection(self):
        try:
            status, collections = self.client.list_collections()
            return collections
        except Exception as e:
            print("Failed to list collections in Milvus: {}".format(e))
            sys.exit(1)


if __name__ == '__main__':
    client = MilvusHelper()
    collection_name = 'test_helper'

    # delete the collection 'test_helper' if it exists
    if client.has_collection(collection_name):
        client.delete_collection(collection_name)

    # create a collection named 'test_helper'
    client.creat_collection(collection_name)

    # get the number of vectors in collection 'test_helper'
    num = client.count(collection_name)
    print(num)

    # Show all collections in Milvus server
    print(client.list_collection())

    # delete the collection
    client.delete_collection(collection_name)
