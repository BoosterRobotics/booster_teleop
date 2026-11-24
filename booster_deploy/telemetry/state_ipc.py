"""Atomic IPC snapshots preserve all ROS fields without pickle or independent recording."""
from collections.abc import Mapping
import os
import threading
import weakref
import msgpack
import numpy as np
from booster_deploy.utils.synced_array import SyncedArray
from .codec import encode

PAYLOAD_CAPACITY=65536
_REQUIRED=frozenset(('state_recv_id','state_recv_ns','low_state_size','low_state_payload',
                     'joint_pos','joint_vel','root_rpy_w','root_ang_vel_b'))

class AtomicLowStateArray(SyncedArray):
    def __init__(self,*args,**kwargs):
        self._frame_lock=threading.RLock()
        self._telemetry_writer=None
        self._telemetry_pid=None
        self._telemetry_last_recv_id=None
        super().__init__(*args,**kwargs)
        ref=weakref.ref(self)
        def child_lock():
            instance=ref()
            if instance is not None and not instance._closed:
                instance._frame_lock=threading.RLock()
                instance._telemetry_writer=None
                instance._telemetry_pid=None
                instance._telemetry_last_recv_id=None
                instance._lock_fd.close()
                instance._lock_fd=open(instance._lock_path,'a+')
        os.register_at_fork(after_in_child=child_lock)

    def write(self,arr):
        with self._frame_lock:
            result=super().write(arr)
        self._record_received(arr)
        return result

    # Filtering happens before anything is copied, so a disabled stage costs one dtype check.
    def _record_received(self,arr):
        if not _REQUIRED<=set(getattr(arr.dtype,'names',None) or ()): return
        if not os.getenv('BOOSTER_TELEMETRY_DIR'): return
        writer=None
        try:
            base=np.asarray(arr)
            row=base.reshape(-1)[0]
            recv_id=int(row['state_recv_id'])
            if recv_id<=0 or recv_id==self._telemetry_last_recv_id: return
            writer=self._writer()
            if not writer.accepts('state'): return
            self._telemetry_last_recv_id=recv_id
            size=max(0,int(row['low_state_size']))
            payload=None
            if size and writer.config.field_enabled('state','low_state'):
                # Zero-copy byte view of the already packed payload inside this record.
                offset=base.dtype.fields['low_state_payload'][1]
                view=base.reshape(-1).view(np.uint8)
                payload=memoryview(view[offset:offset+min(size,view.size-offset)])
            from .state import received
            received(dict(kind='received',state_source='real',recv_id=recv_id,recv_ns=int(row['state_recv_ns']),
                q=np.asarray(row['joint_pos']),dq=np.asarray(row['joint_vel']),
                root_rpy_w=np.asarray(row['root_rpy_w']),root_ang_vel_b=np.asarray(row['root_ang_vel_b']),
                low_state=payload,execution_ack=False,backend='low_state_received'),writer=writer)
        except Exception as exc:
            try:
                from .state import received
                received(None,error=exc,writer=writer if writer is not None else self._writer())
            except Exception: pass

    def _writer(self):
        if self._telemetry_writer is None or self._telemetry_pid!=os.getpid():
            from .runtime import get_writer
            self._telemetry_writer=get_writer()
            self._telemetry_pid=os.getpid()
        return self._telemetry_writer

    def read(self):
        with self._frame_lock: return super().read()


def message_fields(value):
    if value is None or isinstance(value,(str,bytes,bool,int,float,np.ndarray,np.generic)): return value
    if isinstance(value,Mapping): return {str(k):message_fields(v) for k,v in value.items()}
    fields=getattr(value,'get_fields_and_field_types',None)
    if fields is not None: return {name:message_fields(getattr(value,name)) for name in fields()}
    slots=getattr(type(value),'__slots__',())
    if slots:
        names=[]
        for cls in reversed(type(value).__mro__):
            declared=cls.__dict__.get('__slots__',())
            names.extend((declared,) if isinstance(declared,str) else declared)
        return {name.lstrip('_'):message_fields(getattr(value,name)) for name in names if name not in ('__dict__','__weakref__')}
    if hasattr(value,'__dict__'): return {name:message_fields(v) for name,v in vars(value).items()}
    try: return [message_fields(v) for v in value]
    except TypeError as exc: raise TypeError('unsupported ROS field '+str(type(value))) from exc


def pack_message(message):
    payload=msgpack.packb(message_fields(message),use_bin_type=True,default=encode)
    if len(payload)>PAYLOAD_CAPACITY: raise ValueError('low_state snapshot exceeds IPC capacity')
    return payload
