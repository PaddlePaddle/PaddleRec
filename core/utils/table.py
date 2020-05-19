# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class TableMeta(object):
    """
    Simple ParamTable Meta, Contain table_id
    """
    TableId = 1
    
    @staticmethod
    def alloc_new_table(table_id):
        """
        create table with table_id
        Args:
            table_id(int) 
        Return:
            table(TableMeta)  :  a TableMeta instance with table_id
        """
        if table_id < 0:
            table_id = TableMeta.TableId
        if table_id >= TableMeta.TableId:
            TableMeta.TableId += 1
        table = TableMeta(table_id)
        return table

    def __init__(self, table_id):
        """ """
        self._table_id = table_id
