"""Read-only run-time replay keeps raw human, robot feedback and policy simulation distinct."""
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .reader import iter_file
from .run_config import provenance_path, read_execution_config


def timestamp(seconds):
    ms=max(0,round(float(seconds)*1000))
    return f'{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d}.{ms%1000:03d}'


class ReplayRun:
    def __init__(self,directory,*,execution=None,mjcf=None,generated=None,max_gap_s=.5):
        self.directory=Path(directory).resolve()
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        self.origin=int(self.manifest['origin_mono_ns'])
        self.clock=self.manifest.get('clock_domain')
        self.errors=[]
        self.warnings=[]
        self.max_gap_s=float(max_gap_s)
        self.series=defaultdict(list)
        self.transitions=[]
        self.keys=[]
        self.duration=0.
        self._model=None
        self._data=None
        self._manifest=None
        self._geometry=None
        sources=set()
        metadata=self.manifest.get('metadata',{})
        config=read_execution_config(self.directory)
        robot=config.get('robot_config') or {}
        self.mjcf = Path(mjcf).expanduser().resolve() if mjcf else provenance_path(
            self.directory, robot.get('mjcf') or config.get('mjcf') or metadata.get('mjcf'),
            robot.get('mjcf_repo_relative'))
        for path in sorted(self.directory.glob('*.msgpack')):
            for row in iter_file(path,self.errors):
                if row.get('clock_domain') not in (None,self.clock):
                    raise ValueError('mixed clock_domain: refusing cross-clock replay')
                if row.get('origin_mono_ns',self.origin)!=self.origin:
                    raise ValueError('mixed origin_mono_ns: refusing segment re-zeroing')
                ns=row.get('t_mono_ns')
                if ns is None: continue
                seconds=(int(ns)-self.origin)/1e9
                if seconds<0: continue
                self.duration=max(self.duration,seconds)
                stage,data=row['stage'],row['data']
                if stage=='events':
                    if data.get('event')=='execution_metadata': metadata=dict(metadata,**data.get('metadata',{}))
                    state=data.get('changes',{}).get('control_state') if data.get('component')=='ctrl' else None
                    if state is not None: self.transitions.append((seconds,str(state)))
                elif stage=='keys': self.keys.append(dict(seconds=seconds,**data))
                elif stage=='publish':
                    if data.get('human_frame') is not None:
                        self.series['human'].append((seconds,('human',data['human_frame'])))
                    elif data.get('poses') is not None:
                        self.series['human'].append((seconds,('poses',data['poses'])))
                elif stage=='retarget' and data.get('robot_qpos') is not None:
                    self.series['retarget'].append((seconds,('retarget',data['robot_qpos'])))
                elif stage=='state':
                    source=data.get('state_source')
                    if source in ('real','sim'): sources.add(source)
                    if source=='sim' and data.get('qpos') is not None:
                        self.series['sim'].append((seconds,('sim',data['qpos'])))
                    elif source=='real' and data.get('q') is not None:
                        key='received' if data.get('kind')=='received' else 'real'
                        self.series[key].append((seconds,('real',{k:data.get(k) for k in ('q','root_quat_w','root_rpy_w','recv_id')})))
        self.execution=execution or config.get('execution') or metadata.get('execution')
        if self.execution is None and len(sources)==1: self.execution=next(iter(sources))
        if self.execution not in ('real','sim'):
            raise ValueError('missing/ambiguous execution metadata; specify --execution real|sim')
        recorded_topology=config.get('topology') or self.manifest.get('topology') or metadata.get('topology')
        if recorded_topology not in (None, 'unknown', 'retarget'):
            raise ValueError(f'unsupported telemetry topology {recorded_topology!r}; only retarget runs can be replayed')
        self.topology='retarget'
        if self.series.get('received'):
            self.series['real']=self.series.pop('received')
        elif self.execution=='real': self.warnings.append('legacy_consumed_feedback_only')
        self.streams=['human','retarget','sim']+(['real'] if self.execution=='real' else [])
        if generated:
            if self.execution!='real': raise ValueError('sim logs already contain closed-loop state; do not regenerate')
            self._load_generated(generated)
        if self.execution=='real' and not self.series.get('sim'): self.warnings.append('generated_simulation_missing')
        self.transitions.sort()
        self.keys.sort(key=lambda x:x['seconds'])
        self.times={}
        for key,values in self.series.items():
            values.sort(key=lambda x:x[0])
            self.times[key]=[x[0] for x in values]
        self._transition_times=[x[0] for x in self.transitions]
        if self.execution=='real': self.warnings.append('real_translation_unavailable_fixed_display_anchor')
        if self.errors: self.warnings.append('log_integrity_errors')

    def _load_generated(self,path):
        with np.load(path,allow_pickle=False) as values:
            meta=json.loads(str(values['metadata']))
            if meta.get('clock_domain')!=self.clock or meta.get('origin_mono_ns')!=self.origin:
                raise ValueError('generated simulation requires explicit matching run origin and clock_domain')
            for seconds,qpos in zip(values['seconds'],values['qpos']):
                self.series['sim'].append((float(seconds),('sim',qpos.copy())))

    def model(self):
        if self._model is None:
            if not self.mjcf: raise ValueError('missing model metadata; specify --mjcf PATH')
            import mujoco
            from booster_deploy.visualization.web.model import build_model_manifest
            self._model=mujoco.MjModel.from_xml_path(str(Path(self.mjcf).resolve()))
            self._data=mujoco.MjData(self._model)
            self._manifest=build_model_manifest(self._model)
        return self._model

    def model_manifest(self):
        self.model()
        return self._manifest

    def geometry(self):
        from booster_deploy.visualization.web.model import geometry_bundle
        if self._geometry is None: self._geometry=geometry_bundle(self.model())
        return self._geometry

    def describe(self):
        return dict(run_id=self.manifest.get('run_id',self.directory.name),execution=self.execution,
            topology=self.topology,origin_mono_ns=self.origin,clock_domain=self.clock,duration_s=self.duration,
            streams=self.streams,transitions=[dict(seconds=t,state=s) for t,s in self.transitions],
            keys=self.keys,warnings=self.warnings,errors=self.errors,counts={k:len(v) for k,v in self.series.items()},
            recording_mode=self.manifest.get('config',{}).get('recording',{}).get('mode','full_run'))

    def _frame(self,kind,payload,seconds,index):
        from booster_deploy.visualization.web.frames import human_frame_payload,qpos_frame_payload
        from booster_deploy.inputs.pico.coordinate import body_poses_to_human_frame,pico_skeleton_connections
        ns=self.origin+round(seconds*1e9)
        if kind in ('human','poses'):
            frame=body_poses_to_human_frame(np.asarray(payload)) if kind=='poses' else payload
            return human_frame_payload(frame,seq=index,sample_ns=ns,connections=pico_skeleton_connections())
        model=self.model()
        if kind=='real':
            from booster_deploy.robots.t2_contract import CONTROL_JOINT_NAMES
            q=np.asarray(payload['q']).reshape(-1)
            if len(q)!=len(CONTROL_JOINT_NAMES): raise ValueError('real feedback does not match CONTROL_JOINT_NAMES')
            qpos=np.asarray(model.qpos0).copy()
            qpos[:3]=[0,0,1.]
            quat=payload.get('root_quat_w')
            if quat is None:
                rpy=np.asarray(payload['root_rpy_w'],dtype=float)/2
                cr,cp,cy=np.cos(rpy); sr,sp,sy=np.sin(rpy)
                quat=[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]
            qpos[3:7]=quat
            for name,value in zip(CONTROL_JOINT_NAMES,q):
                joint=model.joint(name)
                qpos[int(model.jnt_qposadr[joint.id])]=value
        elif kind=='retarget' and Path(self.mjcf).name=='t2_31dof_revo2hand.xml' and np.size(payload)!=model.nq:
            from booster_deploy.visualization.web.frames import t2_body_qpos
            qpos=t2_body_qpos(model,payload)
        else:
            qpos=np.asarray(payload).reshape(-1)
            if qpos.shape!=(int(model.nq),):
                raise ValueError(f'{kind} qpos layout {qpos.shape} does not match explicit model nq={model.nq}')
        frame=qpos_frame_payload(model,self._data,qpos,seq=index,sample_ns=ns)
        if kind=='real': frame['translation']='fixed_display_anchor_not_world_position'
        return frame

    def snapshot(self,seconds):
        seconds=max(0.,min(float(seconds),self.duration))
        at=bisect_right(self._transition_times,seconds)-1
        state=self.transitions[at][1] if at>=0 else ('simulation' if self.execution=='sim' else 'walking')
        visible=['real'] if self.execution=='real' and state=='walking' else list(self.streams)
        frames={}; missing={}
        for stream in self.streams:
            if stream not in visible:
                missing[stream]='inactive_walking'
                continue
            values=self.series.get(stream,[])
            index=bisect_right(self.times.get(stream,[]),seconds)-1
            if index<0:
                missing[stream]='no_sample'
                continue
            when,(kind,payload)=values[index]
            if seconds-when>self.max_gap_s:
                missing[stream]='recording_gap'
                continue
            try: frames[stream]=self._frame(kind,payload,when,index)
            except (ValueError,KeyError,TypeError) as exc: missing[stream]=str(exc)
        return dict(seconds=seconds,time=timestamp(seconds),state=state,visible_streams=visible,frames=frames,
                    missing=missing,keys=[k for k in self.keys if abs(k['seconds']-seconds)<=.5])
