"""Source progress at the actual consumer, independent of grid/window clocks."""
from collections import OrderedDict, deque


class SourceIdentity:
    """Identity of accepted source events, not a claim of unique device samples.

    Backward counters create a new epoch, but a new stream/explicit epoch never
    inherits the previous stream's highwater/reset counter. Legacy stays unknown.
    """
    def __init__(self):
        self.key = None
        self.last = None
        self.reset_epoch = 0

    def ids(self, frame):
        return tuple(tuple(x) for x in getattr(frame, 'source_ids', ()))


class InputHealth:
    def __init__(self):
        self.highwater = OrderedDict()
        self.events = deque(maxlen=4096)
        self.last_advance_ns = None
        self.known = False
        self.state = 'warmup'

    def observe(self, sources, now_ns, *, state='active'):
        self.state = state
        self.known = any(epoch is not None and seq is not None for epoch, seq in sources)
        if state != 'active':
            return state
        advanced = False
        for epoch, seq in sources:
            if epoch is None or seq is None:
                continue
            if seq > self.highwater.get(epoch, -1):
                self.highwater[epoch] = seq
                self.highwater.move_to_end(epoch)
                advanced = True
        while len(self.highwater) > 32:
            self.highwater.popitem(last=False)
        if advanced:
            self.events.append(int(now_ns))
            self.last_advance_ns = int(now_ns)
        return 'new' if advanced else ('reused' if self.known else 'unknown')

    def facts(self):
        """Owned raw observations only; do not prune, measure, or advance a clock."""
        return dict(events=list(self.events), last_advance_ns=self.last_advance_ns,
                    known=self.known, state=self.state, clock='consumer_local_monotonic')

    def snapshot(self, now_ns):
        while self.events and self.events[0] <= now_ns - 1_000_000_000:
            self.events.popleft()
        return dict(known=self.known, effective_hz=float(len(self.events)) if self.known else None,
                    stall_ms=None if self.last_advance_ns is None else max(0., (now_ns-self.last_advance_ns)/1e6),
                    last_advance_ns=self.last_advance_ns, state=self.state,
                    clock='consumer_local_monotonic')
