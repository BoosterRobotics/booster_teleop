"""NumPy wire arrays preserve dtype, shape and raw bytes."""
import numpy as np

def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('object arrays are not telemetry tensors')
        return {'__ndarray__': True, 'dtype': value.dtype.str, 'shape': list(value.shape), 'data': value.tobytes(order='C')}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def decode(value):
    if value.get('__ndarray__') is True:
        return np.frombuffer(value['data'], dtype=np.dtype(value['dtype'])).reshape(value['shape']).copy()
    return value

def estimate(value):
    if isinstance(value, np.ndarray):
        return value.nbytes + 512
    if hasattr(value, 'detach'):
        return value.numel()*value.element_size()+512
    if isinstance(value, dict):
        return 128+sum(estimate(k)+estimate(v) for k,v in value.items())
    if isinstance(value,(tuple,list)):
        return 128+sum(estimate(v) for v in value)
    if isinstance(value,str):
        return len(value.encode('utf8'))+64
    if isinstance(value,memoryview): return value.nbytes+64
    if isinstance(value,(bytes,bytearray)):
        return len(value)+64
    return 64

def snapshot(value):
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject: raise TypeError('object array')
        return value.copy(order='C')
    if hasattr(value,'detach'):
        return value.detach().cpu().numpy().copy()
    if isinstance(value,dict): return {str(k):snapshot(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [snapshot(v) for v in value]
    if isinstance(value,(bytearray,memoryview)): return bytes(value)
    if value is None or isinstance(value,(str,bytes,int,float,bool,np.generic)): return value
    raise TypeError('unsupported snapshot '+type(value).__name__)
