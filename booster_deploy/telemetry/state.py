"""State snapshots are recorded once at the actual model consumption tick."""
import msgpack
from .runtime import record

def received(facts,*,error=None,writer=None):
    if writer is None:
        try:
            from .runtime import get_writer
            writer=get_writer()
        except Exception: return False
    try:
        if error is not None or facts is None: raise error or RuntimeError('received snapshot unavailable')
        return writer.record('state',facts,t_mono_ns=facts['recv_ns'])
    except Exception as exc:
        if hasattr(writer,'mark_incomplete'):
            try: writer.mark_incomplete('received snapshot: '+repr(exc))
            except Exception: pass
        return False


def consumed(controller, policy_tick, source_ids):
    data=controller.robot.data
    state_id=f'{policy_tick}:state'
    sim=getattr(controller,'mj_data',None) is not None
    facts=dict(state_id=state_id,policy_tick=policy_tick,source_ids=source_ids,state_source='sim' if sim else 'real',
        recv_id=getattr(controller,'state_recv_id',None),recv_ns=getattr(controller,'state_recv_ns',None),
        q=getattr(data,'joint_pos',None),dq=getattr(data,'joint_vel',None),
        root_quat_w=getattr(data,'root_quat_w',None),root_ang_vel_b=getattr(data,'root_ang_vel_b',None),
        root_pos_w=getattr(data,'root_pos_w',None) if sim else None,
        root_lin_vel_b=getattr(data,'root_lin_vel_b',None) if sim else None,execution_ack=False)
    raw=getattr(controller,'_telemetry_low_state',None)
    if raw is not None: facts['low_state']=raw
    if sim:
        facts['imu_sensors']=controller.mj_data.sensordata
        facts['qpos']=controller.mj_data.qpos
        facts['qvel']=controller.mj_data.qvel
    record('state',**facts)
    return state_id
