"""Record only the discrete source frame actually consumed by IK."""
from .runtime import record

def consumed(frame, output, entered_ns, done_ns, sources):
    return record('retarget',kind='discrete',source_channel='retarget_input',source_ids=sources,human_frame=frame.human_frame,
        human_format=frame.human_format,robot_qpos=output,backend_entry_ns=entered_ns,backend_done_ns=done_ns,
        native_seq=frame.seq,source_recv_ns=frame.recv_ns,offline_sample_ns=getattr(frame,"offline_sample_ns",None),
        offline_cycle=getattr(frame,"offline_cycle",0),clock_domain=getattr(frame,"clock_domain",None),
        native_index=getattr(frame,"native_index",None),loop=getattr(frame,"loop",None),stream_id=getattr(frame,"stream_id",None))
