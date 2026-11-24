"""The actual model boundary keeps feeds, actions and targets separate."""
import os
import uuid
from .runtime import record
from .state import consumed

class Encoder:
    def __init__(self):
        self.session=uuid.uuid4().hex
        self.tick=0

    def begin(self, controller, feeds, source_ids=(), window=None, layout=None, source_channel=None):
        if self.tick==0:
            sim=getattr(controller,'mj_data',None) is not None
            record('events',event='execution_metadata',metadata={'execution':'sim' if sim else 'real'})
        self.tick+=1
        tick=f'{self.session}:{self.tick}'
        controller._telemetry_policy_tick=tick
        controller._telemetry_sources=source_ids
        state=consumed(controller,tick,source_ids)
        record('encoder',phase='entry',policy_tick=tick,state_id=state,source_ids=source_ids,
               window=window,feeds=feeds,layout=layout,source_channel=source_channel)
        return tick

    def result(self,tick,raw_action,clipped_action,target,**facts):
        return record('encoder',phase='result',policy_tick=tick,raw_action=raw_action,
                      clipped_action=clipped_action,target=target,execution_ack=False,**facts)

_DEFAULT=object()

def target(controller,physical_target,*,policy_tick=_DEFAULT,**facts):
    if policy_tick is _DEFAULT: policy_tick=getattr(controller,'_telemetry_policy_tick',None)
    return record('encoder',phase='physical_target',policy_tick=policy_tick,
        source_ids=getattr(controller,'_telemetry_sources',()) if policy_tick is not None else (),physical_target=physical_target,execution_ack=False,**facts)
