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

import os
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient


def is_afs_path(path):
    """R 
    """
    if path.startswith("afs") or path.startswith("hdfs"):
        return True
    return False


class LocalFSClient(object):
    """
    Util for local disk file_system io 
    """
    
    def __init__(self):
        """R
        """
        pass
    
    def write(self, content, path, mode):
        """
        write to file
        Args:
            content(string)
            path(string)
            mode(string): w/a  w:clear_write a:append_write
        """
        temp_dir = os.path.dirname(path)
        if not os.path.exists(temp_dir): 
            os.makedirs(temp_dir)
        f = open(path, mode)
        f.write(content)
        f.flush()
        f.close()

    def cp(self, org_path, dest_path):
        """R
        """
        temp_dir = os.path.dirname(dest_path)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        return os.system("cp -r " + org_path + " " + dest_path)

    def cat(self, file_path):
        """R
        """
        f = open(file_path)
        content = f.read()
        f.close()
        return content

    def mkdir(self, dir_name):
        """R
        """
        os.makedirs(dir_name)

    def remove(self, path):
        """R
        """
        os.system("rm -rf " + path)
    
    def is_exist(self, path):
        """R
        """
        if os.system("ls " + path) == 0:
            return True
        return False

    def ls(self, path):
        """R
        """
        files = os.listdir(path)
        return files


class FileHandler(object):
    """
    A Smart file handler. auto judge local/afs by path 
    """
    def __init__(self, config):
        """R
        """
        if 'fs_name' in config:
            hadoop_home="$HADOOP_HOME"
            hdfs_configs = {
                "hadoop.job.ugi": config['fs_ugi'], 
                "fs.default.name": config['fs_name']
            }
            self._hdfs_client = HDFSClient(hadoop_home, hdfs_configs)
        self._local_fs_client = LocalFSClient()

    def is_exist(self, path):
        """R
        """
        if is_afs_path(path):
            return self._hdfs_client.is_exist(path)
        else:
            return self._local_fs_client.is_exist(path)

    def get_file_name(self, path):
        """R
        """
        sub_paths = path.split('/')
        return sub_paths[-1]

    def write(self, content, dest_path, mode='w'):
        """R
        """
        if is_afs_path(dest_path):
            file_name = self.get_file_name(dest_path)
            temp_local_file = "./tmp/" + file_name
            self._local_fs_client.remove(temp_local_file)
            org_content = ""
            if mode.find('a') >= 0:
                org_content = self._hdfs_client.cat(dest_path)
            content = content + org_content
            self._local_fs_client.write(content, temp_local_file, mode) #fleet hdfs_client only support upload, so write tmp file
            self._hdfs_client.delete(dest_path + ".tmp")
            self._hdfs_client.upload(dest_path + ".tmp", temp_local_file)
            self._hdfs_client.delete(dest_path + ".bak")
            self._hdfs_client.rename(dest_path, dest_path + '.bak')
            self._hdfs_client.rename(dest_path + ".tmp", dest_path)
        else:
            self._local_fs_client.write(content, dest_path, mode)
    
    def cat(self, path):
        """R
        """
        if is_afs_path(path):
            hdfs_cat = self._hdfs_client.cat(path)
            return hdfs_cat
        else:
            return self._local_fs_client.cat(path)
    
    def ls(self, path):
        """R
        """
        files = []
        if is_afs_path(path):
            files = self._hdfs_client.ls(path)
            files = [path + '/' + self.get_file_name(fi) for fi in files]  # absulte path
        else:
            files = self._local_fs_client.ls(path)
            files = [path + '/' + fi for fi in files]  # absulte path
        return files
    
    def cp(self, org_path, dest_path):
        """R
        """
        org_is_afs = is_afs_path(org_path)
        dest_is_afs = is_afs_path(dest_path)
        if not org_is_afs and not dest_is_afs:
            return self._local_fs_client.cp(org_path, dest_path)
        if not org_is_afs and dest_is_afs:
            return self._hdfs_client.upload(dest_path, org_path)
        if org_is_afs and not dest_is_afs: 
            return self._hdfs_client.download(org_path, dest_path)
        print("Not Suppor hdfs cp currently")
