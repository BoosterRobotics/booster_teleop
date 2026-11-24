"""Discrete transitions are not periodic state snapshots."""
from .runtime import record

class Transitions:
    def __init__(self,component):
        self.component=component
        self.previous={}

    def update(self,**state):
        changed={k:v for k,v in state.items() if k not in self.previous or self.previous[k]!=v}
        if not changed: return False
        self.previous.update(changed)
        return record('events',event='state_transition',component=self.component,changes=changed)

def event(name,**facts):
    return record('events',event=name,**facts)
