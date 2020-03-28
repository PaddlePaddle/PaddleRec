"""
Define Dataset
"""
import abc
import copy
import yaml
import time
import datetime
import paddle.fluid as fluid
import kagle.utils.kagle_fs as kagle_fs
import kagle.utils.kagle_util as kagle_util


class Dataset(object):
    """
    Dataset Base
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, config):
        """ 
        """
        self._datasets = {}
        self._config = config
    
    @abc.abstractmethod
    def check_ready(self, params):
        """
        check data ready or not
        Return:
            True/False
        """
        pass

    @abc.abstractmethod
    def load_dataset(self, params): 
        """R
        """
        pass
    
    @abc.abstractmethod
    def preload_dataset(self, params): 
        """R
        """
        pass
    
    @abc.abstractmethod
    def release_dataset(self, params): 
        """R 
        """
        pass


class TimeSplitDataset(Dataset):
    """
    Dataset with time split dir.  root_path/$DAY/$HOUR
    """
    def __init__(self, config):
        """
        init data root_path, time_split_interval, data_path_format
        """
        Dataset.__init__(self, config)
        if 'data_donefile' not in config or config['data_donefile'] is None:
            config['data_donefile'] = config['data_path'] + "/to.hadoop.done" 
        self._path_generator = kagle_util.PathGenerator({'templates': [
            {'name': 'data_path', 'template': config['data_path']},        
            {'name': 'donefile_path', 'template': config['data_donefile']}        
        ]})
        self._split_interval = config['split_interval'] # data split N mins per dir
        self._data_file_handler = kagle_fs.FileHandler(config)

    def _format_data_time(self, daytime_str, time_window_mins):
        """ """
        data_time = kagle_util.make_datetime(daytime_str)  
        mins_of_day = data_time.hour * 60 + data_time.minute
        begin_stage = mins_of_day / self._split_interval
        end_stage = (mins_of_day + time_window_mins) / self._split_interval
        if begin_stage == end_stage and mins_of_day % self._split_interval != 0:
            return None, 0

        if mins_of_day % self._split_interval != 0:
            skip_mins = self._split_interval - (mins_of_day % self._split_interval)
            data_time = data_time + datetime.timedelta(minutes=skip_mins)
            time_window_mins = time_window_mins - skip_mins 
        return data_time, time_window_mins
    
    def check_ready(self, daytime_str, time_window_mins):
        """
        data in [daytime_str, daytime_str + time_window_mins] is ready or not
        Args:
            daytime_str: datetime with str format, such as "202001122200" meanings "2020-01-12 22:00"
            time_window_mins(int): from daytime_str to daytime_str + time_window_mins
        Return:
            True/False
        """
        is_ready = True
        data_time, windows_mins = self._format_data_time(daytime_str, time_window_mins)
        while time_window_mins > 0:
            file_path = self._path_generator.generate_path('donefile_path', {'time_format': data_time}) 
            if not self._data_file_handler.is_exist(file_path):
                is_ready = False
                break
            time_window_mins = time_window_mins - self._split_interval
            data_time = data_time + datetime.timedelta(minutes=self._split_interval)
        return is_ready
        
    def get_file_list(self, daytime_str, time_window_mins, node_num=1, node_idx=0):
        """
        data in  [daytime_str, daytime_str + time_window_mins], random shard to node_num, return shard[node_idx]
        Args:
            daytime_str: datetime with str format, such as "202001122200" meanings "2020-01-12 22:00"
            time_window_mins(int): from daytime_str to daytime_str + time_window_mins
            node_num(int): data split shard num
            node_idx(int): shard_idx
        Return:
            list, data_shard[node_idx]
        """
        data_file_list = []
        data_time, windows_mins = self._format_data_time(daytime_str, time_window_mins)
        while time_window_mins > 0:
            file_path = self._path_generator.generate_path('data_path', {'time_format': data_time}) 
            sub_file_list = self._data_file_handler.ls(file_path)
            for sub_file in sub_file_list:
                sub_file_name = self._data_file_handler.get_file_name(sub_file)
                if not sub_file_name.startswith(self._config['filename_prefix']):
                    continue
                if hash(sub_file_name) % node_num == node_idx:
                    data_file_list.append(sub_file)
            time_window_mins = time_window_mins - self._split_interval
            data_time = data_time + datetime.timedelta(minutes=self._split_interval)
        return data_file_list 
        

class FluidTimeSplitDataset(TimeSplitDataset):
    """
    A Dataset with time split for PaddleFluid
    """
    def __init__(self, config):
        """ """
        TimeSplitDataset.__init__(self, config)
    
    def _alloc_dataset(self, file_list):
        """ """
        dataset = fluid.DatasetFactory().create_dataset(self._config['dataset_type'])
        dataset.set_batch_size(self._config['batch_size'])
        dataset.set_thread(self._config['load_thread'])
        dataset.set_hdfs_config(self._config['fs_name'], self._config['fs_ugi'])
        dataset.set_pipe_command(self._config['data_converter'])
        dataset.set_filelist(file_list)
        dataset.set_use_var(self._config['data_vars'])
        #dataset.set_fleet_send_sleep_seconds(2)
        #dataset.set_fleet_send_batch_size(80000)
        return dataset

    def load_dataset(self, params):
        """ """ 
        begin_time = params['begin_time']
        windown_min = params['time_window_min']
        if begin_time not in self._datasets:
            while self.check_ready(begin_time, windown_min) == False:
                print("dataset not ready, time:" + begin_time)
                time.sleep(30)
            file_list = self.get_file_list(begin_time, windown_min, params['node_num'], params['node_idx'])
            self._datasets[begin_time] = self._alloc_dataset(file_list)
            self._datasets[begin_time].load_into_memory()
        else:
            self._datasets[begin_time].wait_preload_done()
        return self._datasets[begin_time]
    
    def preload_dataset(self, params): 
        """ """
        begin_time = params['begin_time']
        windown_min = params['time_window_min']
        if begin_time not in self._datasets:
            if self.check_ready(begin_time, windown_min):
                file_list = self.get_file_list(begin_time, windown_min, params['node_num'], params['node_idx'])
                self._datasets[begin_time] = self._alloc_dataset(file_list)
                self._datasets[begin_time].preload_into_memory(self._config['preload_thread'])
                return True
        return False

    def release_dataset(self, params): 
        """ """
        begin_time = params['begin_time']
        windown_min = params['time_window_min']
        if begin_time in self._datasets:
            self._datasets[begin_time].release_memory()
