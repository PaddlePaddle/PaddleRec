import sys
import time
from abc import ABCMeta, abstractmethod

class Trainer(object):
    __metaclass__=ABCMeta
    def __init__(self, config):
        self._status_processor = {}
        self._context = {'status': 'uninit', 'is_exit': False}
       
    def regist_context_processor(self, status_name, processor):
        self._status_processor[status_name] = processor

    def context_process(self, context):
        if context['status'] in self._status_processor:
            self._status_processor[context['status']](context)
        else:
            self.other_status_processor(context)
    
    def other_status_processor(self, context):
        print('unknow context_status:%s, do nothing' % context['status'])
	time.sleep(60)

    def reload_train_context(self):
        pass

    def run(self):
        while True:
            self.reload_train_context()
            self.context_process(self._context)
            if self._context['is_exit']:
                break
