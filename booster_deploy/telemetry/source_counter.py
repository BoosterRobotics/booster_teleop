"""Run-local mmap counter survives publisher restarts independently of logging."""
import fcntl
import mmap
import os
import struct
import threading

_counters={}
_guard=threading.Lock()

class Counter:
    def __init__(self,directory):
        self.fd=os.open(str(directory/'.source-counter'),os.O_RDWR|os.O_CREAT,0o600)
        fcntl.flock(self.fd,fcntl.LOCK_EX)
        try:
            if os.fstat(self.fd).st_size<8: os.ftruncate(self.fd,8)
            self.memory=mmap.mmap(self.fd,8)
        finally: fcntl.flock(self.fd,fcntl.LOCK_UN)
        self.lock=threading.Lock()

    def next(self):
        with self.lock:
            fcntl.flock(self.fd,fcntl.LOCK_EX)
            try:
                value=struct.unpack_from('<Q',self.memory)[0]+1
                struct.pack_into('<Q',self.memory,0,value)
                return value
            finally: fcntl.flock(self.fd,fcntl.LOCK_UN)


def next_id(directory):
    key=str(directory)
    with _guard:
        if key not in _counters: _counters[key]=Counter(directory)
        counter=_counters[key]
    return counter.next()


def _fork_child():
    global _counters,_guard
    for counter in _counters.values():
        counter.memory.close(); os.close(counter.fd)
    _counters={}; _guard=threading.Lock()

os.register_at_fork(after_in_child=_fork_child)
