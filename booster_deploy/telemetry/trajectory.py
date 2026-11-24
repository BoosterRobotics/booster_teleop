"""Resolve committed offline playbacks against an unlimited run's selection history."""
from bisect import bisect_right
from dataclasses import dataclass


REPORT_STEM = "multi-trajectory"


@dataclass(frozen=True)
class Playback:
    cycle: int
    trajectory_index: int
    trajectory_path: str
    start_ns: int
    played_ns: int
    end_ns: int | None
    input_selection: dict | None


def is_unlimited(config):
    return bool(config.get("offline") and config.get("trajectory_switch") == "unlimited")


def committed_playbacks(config, events):
    """Return None for legacy/once; otherwise return monotonic committed windows.

    Prepared input may precede playback_started. Aborted preparations are not
    windows, but still bound the preceding window (including in full-run logs).
    """
    if not is_unlimited(config):
        return None
    history = config.get("trajectory_history")
    if not isinstance(history, list) or not history:
        raise ValueError("unlimited offline run is missing trajectory_history")
    for item in history:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k]
                                             for k in ("path", "name", "stem")):
            raise ValueError("invalid trajectory_history entry")

    preparations = {}
    boundaries = []
    started = []
    for row in sorted(events, key=lambda row: int(row["t_mono_ns"])):
        data = row["data"]
        stamp = int(row["t_mono_ns"])
        event = data.get("event")
        if event == "offline_prepare":
            cycle = data.get("cycle")
            if type(cycle) is int and cycle > 0:
                preparations[cycle] = stamp
                boundaries.append(stamp)
        elif (event == "state_transition" and data.get("component") == "ctrl"
              and data.get("changes", {}).get("control_state") == "walking"):
            boundaries.append(stamp)
        elif event == "offline_playback_started":
            started.append((stamp, data))
    boundaries = sorted(set(boundaries) | {preparations.get(data.get("cycle"), stamp) for stamp, data in started
                                           if type(data.get("cycle")) is int})

    result = []
    cycles = set()
    for stamp, data in started:
        cycle = data.get("cycle")
        index = data.get("trajectory_index")
        if type(cycle) is not int or cycle <= 0 or cycle in cycles:
            raise ValueError(f"invalid or duplicate offline playback cycle: {cycle!r}")
        if type(index) is not int or not 0 <= index < len(history):
            raise ValueError(f"playback cycle {cycle} has invalid trajectory_index: {index!r}")
        path = history[index]["path"]
        if data.get("trajectory_path") != path:
            raise ValueError(f"playback cycle {cycle} path disagrees with trajectory_history[{index}]")
        selection = data.get("input_selection")
        if selection is not None and (not isinstance(selection, dict) or selection.get("path") != path):
            raise ValueError(f"playback cycle {cycle} input selection disagrees with its trajectory path")
        start = preparations.get(cycle, stamp)
        boundary = bisect_right(boundaries, start)
        end = boundaries[boundary] if boundary < len(boundaries) else None
        if not start <= stamp or (end is not None and stamp >= end):
            raise ValueError(f"playback cycle {cycle} started outside its prepared window")
        if result and (result[-1].end_ns is None or start < result[-1].end_ns):
            raise ValueError(f"playback cycle {cycle} overlaps the preceding playback")
        cycles.add(cycle)
        result.append(Playback(cycle, index, path, start, stamp, end, selection))
    return result


def playback_at(playbacks, starts, stamp):
    index = bisect_right(starts, stamp) - 1
    if index < 0:
        return None
    play = playbacks[index]
    return play if play.end_ns is None or stamp < play.end_ns else None
