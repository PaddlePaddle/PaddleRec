"""
Define A Trainer Base
"""
import abc
import time


class Trainer(object):
    """R
    """   
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        """R
        """
        self._status_processor = {}
        self._context = {'status': 'uninit', 'is_exit': False}
       
    def regist_context_processor(self, status_name, processor):
        """
        regist a processor for specify status
        """
        self._status_processor[status_name] = processor

    def context_process(self, context):
        """
        select a processor to deal specify context
        Args:
            context : context with status
        Return:
            None : run a processor for this status
        """
        if context['status'] in self._status_processor:
            self._status_processor[context['status']](context)
        else:
            self.other_status_processor(context)
    
    def other_status_processor(self, context):
        """
        if no processor match context.status, use defalut processor
        Return:
            None, just sleep in base
        """
        print('unknow context_status:%s, do nothing' % context['status'])
        time.sleep(60)

    def reload_train_context(self):
        """
        context maybe update timely, reload for update
        """
        pass

    def run(self):
        """
        keep running by statu context.
        """
        while True:
            self.reload_train_context()
            self.context_process(self._context)
            if self._context['is_exit']:
                break
