"""Off-process MuJoCo state logging.

History of this file:

1. The simulator kept one Python list per field and, every 100 control steps,
   ``np.stack``-ed the whole history and rewrote the whole ``.npz`` **inside the
   control loop**.  Measured on the T2 board: 144 ms at 29k steps, 331 ms at 60k,
   producing ``tick gap`` spikes of ~1 s and an inference rate of 20-40 Hz.
2. Moving the write to a *thread* removed the deep stalls (>500 ms fell by 94%)
   but the writer still shared the interpreter: ``np.savez`` of a 16 MB history
   holds the GIL in bursts, and a residual 100-370 ms stall remained every ~8 s.
3. This version hands the blocks to a **separate process**.  The control thread
   only stores one row into a small staging buffer and puts a ~55 KB block on a
   multiprocessing queue; nothing on the hot path serialises or writes.

File semantics are unchanged: the same ``.npz``, the same field names, float32,
row order, updated every ``flush_every`` steps, with the same operator-facing
``saved ... at N steps`` line (printed by the writer process, so it still lands in
the deploy pane).
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import traceback
from pathlib import Path
from typing import Mapping

import numpy as np


class MujocoStateRecorder:
    """Row-staged recorder whose ``npz`` writer lives in its own process."""

    def __init__(
        self,
        path: str | Path,
        dims: Mapping[str, int],
        *,
        flush_every: int = 100,
        initial_rows: int = 2048,
        max_blocks: int = 512,
        start_timeout: float = 10.0,
    ) -> None:
        self.path = Path(str(path)).with_suffix(".npz")
        self.dims = {str(k): int(v) for k, v in dims.items()}
        self.flush_every = max(1, int(flush_every))
        self.rows = 0
        self.dropped_blocks = 0
        self.last_error: str | None = None
        self.last_saved_rows = 0
        self._queued_rows = 0
        self._closed = False

        self._staging: dict[str, np.ndarray] = {
            name: np.zeros((self.flush_every, dim), dtype=np.float32)
            for name, dim in self.dims.items()
        }
        self._staged = 0

        ctx = mp.get_context("fork") if _can_fork() else mp.get_context("spawn")
        self._ctx = ctx
        self._blocks = ctx.Queue(maxsize=max(1, int(max_blocks)))
        self._acks = ctx.Queue()
        self._proc = ctx.Process(
            target=_writer_main,
            args=(str(self.path), self.dims, self._blocks, self._acks, int(initial_rows)),
            name="mujoco-state-writer",
            daemon=True,
        )
        self._proc.start()
        self._start_timeout = float(start_timeout)
        self._proc.join(0.05)
        if not self._proc.is_alive() and self._proc.exitcode not in (None, 0):
            self.last_error = f"writer process exited with code {self._proc.exitcode}"

    # ------------------------------------------------------------------ control side
    def append(self, values: Mapping[str, np.ndarray]) -> None:
        """Record one step: one row store per field, never a serialisation."""
        if self._closed:
            return
        staged = self._staged
        for name, dim in self.dims.items():
            row = np.asarray(values[name], dtype=np.float32).reshape(-1)
            if row.size != dim:
                raise ValueError(f"{name}: expected {dim} values, got {row.size}")
            self._staging[name][staged] = row
        self._staged = staged + 1
        self.rows += 1
        if self._staged >= self.flush_every:
            self._flush_block()

    def _flush_block(self) -> None:
        staged = self._staged
        if not staged:
            return
        block = {name: buf[:staged].copy() for name, buf in self._staging.items()}
        self._staged = 0
        if not self._proc.is_alive():
            self.last_error = "writer process is not running; state logging disabled"
            self.dropped_blocks += 1
            return
        try:
            self._blocks.put(block, block=False)
            self._queued_rows += staged
        except queue_mod.Full:
            # Never block the control loop; the operator sees the loss counter.
            self.dropped_blocks += 1

    # ------------------------------------------------------------------ lifecycle
    def _drain_acks(self) -> None:
        while True:
            try:
                tag, value = self._acks.get(block=False)
            except queue_mod.Empty:
                return
            if tag == "saved":
                self.last_saved_rows = int(value)
            elif tag == "error":
                self.last_error = str(value)
            elif tag == "rows":
                self.last_saved_rows = int(value)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Wait until the writer process has persisted every queued block."""
        import time

        deadline = time.monotonic() + max(float(timeout), 0.0)
        while time.monotonic() < deadline:
            self._drain_acks()
            if self.last_saved_rows >= self._queued_rows:
                return True
            time.sleep(0.02)
        return False

    def close(self, timeout: float = 15.0) -> None:
        """Flush the partial window, let the writer persist it, then stop it."""
        if self._closed:
            return
        self._closed = True
        self._flush_block()
        try:
            self._blocks.put(None, timeout=5.0)
        except queue_mod.Full:
            pass
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2.0)
            self.last_error = self.last_error or "writer process did not exit in time"
        self._drain_acks()
        try:
            self._blocks.close()
            self._acks.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "MujocoStateRecorder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# --------------------------------------------------------------------------- child


def _writer_main(path: str, dims: Mapping[str, int], blocks, acks, initial_rows: int) -> None:
    """Writer process: owns the only full history and does every ``np.savez``."""
    history = {
        str(name): np.zeros((max(1, int(initial_rows)), int(dim)), dtype=np.float32)
        for name, dim in dims.items()
    }
    rows = 0
    capacity = max(1, int(initial_rows))

    def emit(tag: str, value) -> None:
        try:
            acks.put((tag, value), block=False)
        except queue_mod.Full:
            pass

    while True:
        try:
            block = blocks.get()
        except (EOFError, OSError):
            return
        except KeyboardInterrupt:  # pragma: no cover
            return
        if block is None:
            break
        incoming = int(next(iter(block.values())).shape[0])
        needed = rows + incoming
        if needed > capacity:
            while capacity < needed:
                capacity *= 2
            grown = {}
            for name, dim in dims.items():
                arr = np.zeros((capacity, int(dim)), dtype=np.float32)
                arr[:rows] = history[name][:rows]
                grown[str(name)] = arr
            history = grown
        for name in dims:
            history[str(name)][rows:needed] = block[name]
        rows = needed
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **{k: v[:rows] for k, v in history.items()})
            print(f"saved {path} at {rows} steps", flush=True)
            emit("saved", rows)
        except Exception as exc:  # noqa: BLE001 - never take down the simulator
            emit("error", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
    emit("rows", rows)


def _can_fork() -> bool:
    import sys

    return sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods()
