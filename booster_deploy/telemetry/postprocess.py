"""Synthetic identities never occupy the source-frame namespace."""
import uuid
from .runtime import record

def identity(kind, origins, *, weight=None, parameters=None, token=None,source_channel=None):
    origins=list(origins)
    channels={v.get('source_channel') for v in origins if isinstance(v,dict)}-{None,'unknown'}
    if source_channel is None and len(channels)==1: source_channel=next(iter(channels))
    return dict(kind=kind,id=token or 'synthetic:'+uuid.uuid4().hex,origins=origins,
                weight=weight,parameters=parameters,source_channel=source_channel)

def output(identity,**facts):
    return record('postprocess',identity=identity,source_channel=identity.get('source_channel'),**facts)


def reference_window(frames,kind,measurement,*,live_parent=None):
    from dataclasses import replace
    parent=dict(kind='measurement',**measurement)
    sources=live_parent.source_ids if live_parent is not None else ()
    origins=[parent,live_parent.identity if live_parent is not None else dict(kind='configuration',name='default_joint_pos')]
    result=[]
    for frame in frames:
        lineage=identity(kind,origins)
        output(lineage,source_ids=sources,joint_pos=frame.joint_pos,joint_vel=frame.joint_vel,
            root_pos=frame.root_pos,root_quat=frame.root_quat)
        result.append(replace(frame,identity=lineage,source_ids=sources))
    return result


def window(frames,**parameters):
    sources=tuple(dict.fromkeys(s for frame in frames for s in frame.source_ids))
    lineage=identity('window',[f.identity for f in frames],parameters=parameters)
    lineage['positions']=[dict(position=i,identity=f.identity,source_ids=f.source_ids) for i,f in enumerate(frames)]
    output(lineage,source_ids=sources)
    return lineage,sources
