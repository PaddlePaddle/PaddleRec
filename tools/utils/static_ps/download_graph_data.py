#!/usr/bin/python
""" download graph data """
import os
import sys
import math
import time
import json
import subprocess
import multiprocessing
from multiprocessing import Process
from multiprocessing import Lock
from multiprocessing import Value
from multiprocessing import Array
from multiprocessing import Queue
from threading import Timer

RETRY_TIMES_MAX = 10
WAIT_TIME_OF_DOWNLOAD = 7200  # Seconds

def kill_proc(process, timer_shell):
    """
    kill proc
    """
    print("Timeout, the proc will be killed, wait_seconds[%d]",
            timer_shell.wait_seconds())
    process.kill()
    timer_shell.set_timeout(True)


class TimerShell(object):
    """
    TimerShell
    """
    def __init__(self, wait_seconds):
        """
        Args:
            wait_seconds (float):
        
        """
        self._wait_seconds = wait_seconds
        self._timeout = False

    def run(self, cmd):
        """
        run
        """
        self._timeout = False
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        # print("TimerShell::run() cmd: %s" % cmd)
        timer = Timer(self._wait_seconds, kill_proc, [proc, self])
        try:
            timer.start()
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print('Fail to cmd[' + cmd + '] proc_stderr[' + stderr + ']')
                return -1
        except Exception as e:
            print("catch exception %s", e)
            return -1
        finally:
            timer.cancel()

        if self._timeout:
            self._wait_seconds *= 2
            return -1

        return 0

    def wait_seconds(self):
        """
        wait_seconds
        """
        return self._wait_seconds

    def set_timeout(self, value):
        """
        set timeout
        """
        self._timeout = value
def pop_file(progress_file_name, lock):
    """
    pop file
    """
    lock.acquire()
    file_list = []
    with open(progress_file_name) as f:
        file_list = json.load(f)
    if (len(file_list) == 0):
        lock.release()
        return None, 0
    file = file_list[0]
    del file_list[0]
    with open(progress_file_name, 'w') as f:
        json.dump(file_list, f)
    lock.release()
    return file, len(file_list)

def list_files(path):
    """ list files """
    cmd = hadoop_fs + " -lsr " + path \
            + " | grep -v drwxrwxrwx | awk '{printf(\"%s %s\\n\", $5, $8);}' | sort -nr | awk '{print $2}'"
    print("cmd %s" % cmd)
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    lines = pipe.read()
    if (len(lines) == 0):
        print("list_files empty")
        exit(-1)
    return lines.decode().strip().split('\n')

def init(progress_file_name, src_hadoop_path, dst_local_path):
    """ init """
    sys.stdout.flush()
    os.system("rm -rf %s > /dev/null 2>&1" % dst_local_path)
    os.system("mkdir -p %s > /dev/null 2>&1" % dst_local_path)
    hdfs_file_list = list_files(src_hadoop_path)
    src_hadoop_path_without_prefix = src_hadoop_path.replace('afs:', '')
    src_hadoop_path_without_prefix = src_hadoop_path_without_prefix.replace('//', '/')
    new_file_list = []
    for file in hdfs_file_list:
        file = file.replace('//', '/')
        new_file = file.replace(src_hadoop_path_without_prefix, '')
        if new_file[0] == '/':
            new_file = new_file[1:]
        new_file_list.append(new_file)
    with open(progress_file_name, 'w') as f:
        json.dump(new_file_list, f)

def fini(progress_file_name):
    """ fini operator """
    os.system('rm -rf ' + progress_file_name)
    print("download and shard model finished")
    sys.stdout.flush()

def excute_func(tid, lock, running_task_num, timer_shell, is_subprocess_running, exception_queue,
        src_hadoop_path, dst_local_path, progress_file_name):
    """ execute func """
    is_subprocess_running[tid] = 1
    try:
        while True:
            file, remain = pop_file(progress_file_name, lock) 
            if not file:
                break
        
            print("[%s][%d] Downloading file[%s], remain %s..." % (time.strftime('%Y-%m-%d %H:%M:%S', \
                time.localtime(time.time())), tid, file, remain))
            running_task_num.value += 1
            local_file = dst_local_path + '/' + file
            file_path = os.path.dirname(local_file)
            os.system("mkdir -p %s > /dev/null 2>&1" % file_path)
            if os.path.isfile(local_file):
                os.remove(local_file)

            cmd = hadoop_fs + " -get %s/%s %s" % (src_hadoop_path, file, local_file)
        
            retry_times = 0
            while True:
                ret = timer_shell.run(cmd)
                if ret == 0:
                    break
        
                if (retry_times > RETRY_TIMES_MAX):
                    print("[%s][%d] download file[%s] too many failed" % (time.strftime('%Y-%m-%d %H:%M:%S', \
                            time.localtime(time.time())), tid, file))
                    raise Exception
                    exit(-1)
        
                if os.path.isfile(local_file):
                    os.remove(local_file)
        
                print("[%s][%d] download file[%s] failed, retry_times[%d], running task %d, t[%d]" \
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            tid, file, retry_times, running_task_num.value, tid))
                retry_times += 1
        
            running_task_num.value -= 1
            print("[%s][%d] Downloaded file[%s], running task %d" % (time.strftime('%Y-%m-%d %H:%M:%S', \
                time.localtime(time.time())), tid, file, running_task_num.value))
            sys.stdout.flush()
    except Exception as e:
        print('download exception')
        exception_queue.put(e)

if __name__ == '__main__':
    src_hadoop_path = sys.argv[1]
    fs_name = sys.argv[2]
    fs_ugi = sys.argv[3]
    hadoop_client_path=sys.argv[4]
    dst_local_path = sys.argv[5]
    hadoop_fs = "%s/hadoop/bin/hadoop fs -Dfs.default.name=%s -Dhadoop.job.ugi=%s" % \
            (hadoop_client_path, fs_name, fs_ugi)
    print("input graph data path: %s" % src_hadoop_path)
    cpu_num = multiprocessing.cpu_count()
    pid = os.getpid()
    progress_file_name = '.download_locker.' + str(pid)
    init(progress_file_name, src_hadoop_path, dst_local_path)
    process_list = []
    lock = Lock()
    running_task_num = Value('i', 0)
    is_subprocess_running = Array('i', [0] * cpu_num)
    exception_queue = Queue()
    for i in range(0, cpu_num):
        timer_shell = TimerShell(WAIT_TIME_OF_DOWNLOAD)
        p = Process(target=excute_func, args=(i, lock, running_task_num, timer_shell,
                                            is_subprocess_running, exception_queue, src_hadoop_path,
                                            dst_local_path, progress_file_name))
        p.start()
        process_list.append(p)

    for i in range(0, cpu_num):
        p = process_list[i]
        print("Joining download thread, t[%d]", i)
        print("thread alive %d, t[%d]" % (p.is_alive(), i))
        if is_subprocess_running[i] != 1:
            print("download thread not start, wait 30s, t[%d]", i)
            time.sleep(30)
            print("download thread stat %d, t[%d]" % (is_subprocess_running[i], i))
        if is_subprocess_running[i] == 1:
            print("joining thread, t[%d]", i)
            p.join()
        else:
            print("terminate thread, t[%d]", i)
            p.terminate()
            #print("joining thread, t[%d]", i)
            #p.join()
        print("Joined download thread, t[%d]", i)

    if not exception_queue.empty():
        print("Fail to download %s to %s" % (src_hadoop_path, dst_local_path))
        exit(-1)

    print("Downloaded %s to %s" % (src_hadoop_path, dst_local_path))
    sys.stdout.flush()

    fini(progress_file_name)
    os.system("rm -rf core.* > /dev/null 2>&1 &")
    os.system("rm -rf hs_err_* > /dev/null 2>&1 &")

    exit(0)
