"""Cycle-scoped entry acknowledgement and out-of-loop input acknowledgements."""
class EntryHandshake:
    def __init__(self, cycle):
        self.cycle = int(cycle)
        self.window = None
        self.output = None
        self.cancelled = False

    def prepared(self, window):
        if not self.cancelled and self.window is None and window:
            self.window = list(window)

    def complete(self, session, tick):
        if self.cancelled or self.window is None or not session:
            return False
        self.output = (str(session), int(tick))
        return True

    def accept_published(self, cycle, session, tick):
        return (not self.cancelled and self.output is not None
                and int(cycle) == self.cycle and self.output == (str(session), int(tick)))

    def cancel(self):
        self.cancelled = True
        self.output = None


def record_startup(policy, cycle):
    if not cycle:
        return
    def values(value):
        if hasattr(value, 'detach'):
            value=value.detach().cpu().numpy()
        return value.tolist()
    names=('input_transition_s','resume_transition_s','rt_reference_transition_s',
        'rt_reference_align_heading','rt_reference_align_xy','rt_reference_align_z',
        'action_offset_source','action_scale_factor','pico_input_mode','pico_human_format')
    policy._diag.event('offline_startup_snapshot', dict(cycle=int(cycle),
        state=dict(joint_pos=values(policy.robot.data.joint_pos),quat=values(policy.robot.data.root_quat_w)),
        policy={key:getattr(policy.cfg,key) for key in names if hasattr(policy.cfg,key)}))


def acknowledge_first(cycle, name='replay.first'):
    import os
    import threading
    from booster_deploy.utils.replay_control import _write_token
    event = threading.Event()
    directory = os.getenv('BOOSTER_DIAG_DIR')
    def send():
        if event.wait(45.):
            _write_token(directory, name, cycle)
    threading.Thread(target=send, name='offline-first-ack', daemon=True).start()
    return event
