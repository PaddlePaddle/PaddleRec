import copy
import yaml
from abc import ABCMeta, abstractmethod

class TableMeta:
    TableId = 1
    
    @staticmethod
    def alloc_new_table(table_id):
        if table_id < 0:
            table_id = TableMeta.TableId
        if table_id >= TableMeta.TableId:
            TableMeta.TableId += 1
        table = TableMeta(table_id)
        return table

    def __init__(self, table_id):
        self._table_id = table_id
        pass
