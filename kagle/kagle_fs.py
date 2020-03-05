import os
import time
from paddle.fluid.incubate.fleet.utils.hdfs import HDFSClient

def is_afs_path(path):
    if path.startswith("afs") or path.startswith("hdfs"):
        return True
    return False

class LocalFSClient:
    def __init__(self):
        pass
    def write(self, content, path, mode):
        temp_dir = os.path.dirname(path)
        if not os.path.exists(temp_dir): 
            os.makedirs(temp_dir)
        f = open(path, mode)
        f.write(content)
        f.flush()
        f.close()

    def cp(self, org_path, dest_path):
        temp_dir = os.path.dirname(dest_path)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        return os.system("cp -r " + org_path + " " + dest_path)

    def cat(self, file_path):
        f = open(file_path)
        content = f.read()
        f.close()
        return content

    def mkdir(self, dir_name):
        os.system("mkdir -p " + path)

    def remove(self, path):
        os.system("rm -rf " + path)
    
    def is_exist(self, path):
        if os.system("ls " + path) == 0:
            return True
        return False

    def ls(self, path):
        files = os.listdir(path)
        files = [ path + '/' + fi for fi in files ]
        return files

class FileHandler:
    def __init__(self, config):
        if 'fs_name' in config:
            hadoop_home="$HADOOP_HOME"
            hdfs_configs = {
                "hadoop.job.ugi": config['fs_ugi'], 
                "fs.default.name": config['fs_name']
            }
            self._hdfs_client = HDFSClient(hadoop_home, hdfs_configs)
        self._local_fs_client = LocalFSClient()

    def is_exist(self, path):
        if is_afs_path(path):
            return self._hdfs_client.is_exist(path)
        else:
            return self._local_fs_client.is_exist(path)

    def get_file_name(self, path):
        sub_paths = path.split('/')
        return sub_paths[-1]

    def write(self, content, dest_path, mode='w'):
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
        if is_afs_path(path):
            print("xxh go cat " + path)
            hdfs_cat = self._hdfs_client.cat(path)
            print(hdfs_cat)
            return hdfs_cat
        else:
            return self._local_fs_client.cat(path)
    
    def ls(self, path):
        if is_afs_path(path):
            return self._hdfs_client.ls(path)
        else:
            return self._local_fs_client.ls(path)

    
    def cp(self, org_path, dest_path):
        org_is_afs = is_afs_path(org_path)
        dest_is_afs = is_afs_path(dest_path)
        if not org_is_afs and not dest_is_afs:
            return self._local_fs_client.cp(org_path, dest_path)
        if not org_is_afs and dest_is_afs:
            return self._hdfs_client.upload(dest_path, org_path)
        if org_is_afs and not dest_is_afs: 
            return self._hdfs_client.download(org_path, dest_path)
        print("Not Suppor hdfs cp currently")
            
