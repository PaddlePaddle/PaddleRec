"""
Construct ParamTable Meta 
"""
import copy
import yaml


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
