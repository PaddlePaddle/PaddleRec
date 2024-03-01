# -*- coding=utf-8 -*-
""" 
create date: 2023/03/30
update date: 2023/03/30
__author__ : yangjunchao@baidu.com
"""
import multiprocessing
import logging
import random
import copy
import time
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Pool
import sys
import os
import copy

LOG_FILE = "del_zero_token.log"
logger = logging.getLogger('del_zero_token')
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(levelname)s: %(asctime)s %(process)d'
                        ' [%(filename)s:%(lineno)s][%(funcName)s] %(message)s')
debug_handler = logging.FileHandler(LOG_FILE, 'a')
debug_handler.setFormatter(fmt)
debug_handler.setLevel(logging.DEBUG)
logger.addHandler(debug_handler)
print("the del zero token log will be saved in %s" % LOG_FILE)

CPU_NUM = multiprocessing.cpu_count()
"""
del zero token
"""

def my_multi_base(function, queue, lock):
    """
      function name
      queue  task queue
      lock
    """
    while True:
        if queue.empty() == True:
            return True
        lock.acquire()
        row = ""
        if queue.empty() == True:
            lock.release()
            return True
        row = queue.get(False)
        lock.release()
        if type(row) == dict:
            function(**row)
        else:
            function(*row)


def easy_task_scheduler(function, thread_num, parameter_list):
    """
    function
    thread_num
    parameter_list
    """
    manager = multiprocessing.Manager()
    q = manager.Queue()
    lock = manager.Lock()
    process_list = []
    for row in parameter_list:
        if type(row) in (tuple, dict, list):
            q.put(row)
        else:
            logger.warning("Parameter mismatch: parameter needs to be a dictionary or array")
            return False
    for i in range(thread_num):
        p = multiprocessing.Process(target = my_multi_base, args=(function, q, lock))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()
    return True
    

def deal_token(work_dir, file_name):
    """
    del zero token
    """
    ori_file_name = os.path.join(work_dir, file_name)
    temp_file_name = os.path.join(work_dir, file_name + "_temp")
    file = open(ori_file_name, 'r')
    file_new = open(temp_file_name, 'w')
    while True:
        line = file.readline()
        if line:
            line_list = line.strip().split('\t')
            length = len(line_list)
            for i in range(length):
                if len(line_list[i]) >= 5 and line_list[i][0:5] == "9108:":
                   pos = line_list[i].find(",0")
                   if pos > 0:
                       line_list[i] = line_list[i][0: pos]
                   break
            new_line = "\t".join(line_list)
            file_new.write(new_line + "\n")
        else:
            break
    file.close()
    file_new.close()
    cmd = "mv %s %s" % (temp_file_name, ori_file_name)
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError("Fail to run cmd[%s] ret[%d]" % (cmd, ret))
    logger.info("file %s done" % ori_file_name)
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("len(sys.argv) should >=3")
        exit(-1)
    work_dir = sys.argv[1]  
    ntype2files = sys.argv[2]
    ntype2file_list = ntype2files.split(",")
    dir_list = []
    for ntype2file in ntype2file_list:
        dir_list.append(ntype2file.split(":")[1].strip())
    dir_list = list(set(dir_list))
    real_dir_list = []
    for dir_name in dir_list:
        work_path = os.path.join(work_dir, dir_name)
        if os.path.exists(work_path) is False:
            logger.warning("work_path %s not exit" % work_path)
        else:
            real_dir_list.append(dir_name)
    if len(real_dir_list) == 0:
        logger.warning("all work_path not exit")
        exit(0)
    work_task = []
    for dir_name in real_dir_list:
        work_path = os.path.join(work_dir, dir_name)
        files = os.listdir(work_path)
        for file in files:
            work_task.append([work_path, file])
    easy_task_scheduler(deal_token, CPU_NUM, work_task)
    logger.info("task done")
