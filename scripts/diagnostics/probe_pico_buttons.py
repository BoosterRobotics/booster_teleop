#!/usr/bin/env python3
"""Print what the PICO controller actually reports on ``pico_rt.raw``.

Use this when a recording chord fires at the wrong moment: it shows the raw
button bitmask and trigger values per distinct combination, so a vendor mapping
problem (e.g. a trigger/alias sharing a face-button bit) is visible immediately.

The probe is deliberately chatty about *silence*: if no packets arrive it says
so, because "no output" usually means the publisher is not running (the raw
stream only exists while ``pico_rt_zmq_publisher`` is up with ``--raw-bind``).

    python scripts/diagnostics/probe_pico_buttons.py                 # 15 s
    python scripts/diagnostics/probe_pico_buttons.py --seconds 30
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import zmq

from booster_deploy.inputs.pico.raw_stream import decode_pico_raw
from booster_deploy.types.pico_types import PicoButtonBits


def _names(bits: int) -> str:
    parts = [flag.name for flag in PicoButtonBits if flag.value and bits & flag.value]
    return "|".join(parts) if parts else "NONE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5581")
    parser.add_argument("--topic", default="pico_rt.raw")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--triggers", type=float, default=0.05,
                        help="round trigger values to this step before grouping")
    args = parser.parse_args()

    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 512)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
    socket.connect(args.endpoint)

    combos: Counter[tuple[int, int, int]] = Counter()
    kinds: Counter[str] = Counter()
    packets = 0
    rejected = 0
    without_control = 0
    step = max(args.triggers, 1e-3)
    start = time.monotonic()
    deadline = start + args.seconds
    next_beat = start + 1.0

    print(f"[probe] listening on {args.endpoint}/{args.topic} for {args.seconds:.0f} s", flush=True)
    print("[probe] hold the recording modifier (both triggers) and press X, then Y, then release",
          flush=True)
    while time.monotonic() < deadline:
        if not socket.poll(200):
            now = time.monotonic()
            if now >= next_beat:
                next_beat = now + 1.0
                print(f"[probe] t={now - start:4.1f}s packets={packets} kinds={dict(kinds)} "
                      f"rejected={rejected} no_control={without_control}", flush=True)
                if packets == 0 and now - start > 3.0:
                    print("[probe] no packets on this topic: is the teleop session running "
                          "and was the publisher started with --raw-bind tcp://0.0.0.0:5581 ?",
                          flush=True)
            continue
        packets += 1
        parts = socket.recv_multipart()
        try:
            packet = decode_pico_raw(parts[-1])
        except ValueError as exc:
            rejected += 1
            if rejected == 1:
                print(f"[probe] decode error on first packet: {exc}", flush=True)
            continue
        kind = str(packet.get("kind"))
        kinds[kind] += 1
        values = packet.get("raw_control") or packet.get("routed_control")
        if not isinstance(values, dict):
            without_control += 1
            continue
        bits = int(values.get("buttons", 0))
        left = round(float(values.get("left_trigger", 0.0)) / step) * step
        right = round(float(values.get("right_trigger", 0.0)) / step) * step
        combos[(bits, int(left * 100), int(right * 100))] += 1

    socket.close(linger=0)
    print(f"[probe] done: packets={packets} kinds={dict(kinds)} rejected={rejected} "
          f"no_control={without_control}", flush=True)
    if not combos:
        print("[probe] no control packets seen — cannot infer the button map", flush=True)
        return 1
    print("[probe] distinct control states (count, buttons, triggers):", flush=True)
    for (bits, left, right), count in combos.most_common(24):
        print(f"  n={count:<5d} buttons={bits:<3d} [{_names(bits):<28}] "
              f"L={left / 100:.2f} R={right / 100:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
