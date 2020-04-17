from multiprocessing import Queue
import threading
import time


class ProgressBar(threading.Thread):

    def __init__(self, tqdm_fn, queue: Queue, total: int,
                 position: int, desc: str):
        super(ProgressBar, self).__init__()
        self._should_stop = threading.Event()
#        print('', end='', flush=True)
        self.tqdm = tqdm_fn(total=total, position=position, desc=desc)
#        print('', end='', flush=True)
        self.tqdm.update(0)
        self.queue = queue

    def stop(self):
        self._should_stop.set()

    def stopped(self):
        return self._should_stop.isSet()

    def run(self):
        while True:
            if self.stopped():
                self.tqdm.close()
                return
            if self.queue.empty():
                time.sleep(1)
                continue
            to_add = self.queue.get(timeout=1)
            self.tqdm.update(to_add)
