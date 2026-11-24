"""Every accepted key request/receive fact is unthrottled."""
from .runtime import record

def key(value, *, phase, origin, **facts):
    return record('keys',key=str(value),phase=phase,origin=origin,**facts)
