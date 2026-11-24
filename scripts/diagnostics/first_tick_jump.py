#!/usr/bin/env python3
"""First-tick jump check: how far are joint targets from measurements when the policy takes over tracking?

A suspected cause of falls after switching tasks (rt → sen) is "no entry transition on the first takeover tick": the new policy's first
``dof_targets`` are sent directly while the robot is still standing. This script quantifies that mismatch using simulation logs in
``logs/mujoco_state/teleop_*.npz`` (containing ``joint_pos`` and ``dof_targets``):

* ``first_max``  First-tick max |target - measured| (degrees)
* ``first_rms``  First-tick RMS (degrees)
* ``settle_max`` Steady-state values after tick 50 (reference: following error during normal tracking)

Usage:
    python3 scripts/diagnostics/first_tick_jump.py logs/mujoco_state/xxx.npz
    python3 scripts/diagnostics/first_tick_jump.py <npz> --rows 80 --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEG = 180.0 / np.pi


def load(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        if "joint_pos" not in data or "dof_targets" not in data:
            raise SystemExit(f"{npz_path} lacks joint_pos/dof_targets; it may not be a state log written by deploy")
        measured = np.asarray(data["joint_pos"], dtype=np.float64)
        targets = np.asarray(data["dof_targets"], dtype=np.float64)
    if measured.shape != targets.shape:
        raise SystemExit(f"Shape mismatch: joint_pos {measured.shape} vs dof_targets {targets.shape}")
    return measured, targets


def summarise(measured: np.ndarray, targets: np.ndarray, rows: int) -> dict:
    delta = np.abs(targets - measured) * DEG
    per_row_max = delta.max(axis=1)
    per_row_rms = np.sqrt((delta ** 2).mean(axis=1))
    head = min(rows, len(per_row_max))
    settle_from = min(50, max(0, len(per_row_max) - 1))
    settle = per_row_max[settle_from:]
    worst_joint = int(np.argmax(delta[0])) if len(delta) else -1
    return {
        "rows": int(len(per_row_max)),
        "first_max_deg": float(per_row_max[0]),
        "first_rms_deg": float(per_row_rms[0]),
        "first_joint": worst_joint,
        "head_max_deg": [round(float(v), 2) for v in per_row_max[:head]],
        "settle_median_deg": float(np.median(settle)) if len(settle) else None,
        "settle_p90_deg": float(np.percentile(settle, 90)) if len(settle) else None,
        "global_max_deg": float(per_row_max.max()) if len(per_row_max) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Joint-target jump on the first takeover tick")
    parser.add_argument("npz", type=Path)
    parser.add_argument("--rows", type=int, default=20, help="Number of initial ticks to print (default: 20)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    measured, targets = load(args.npz)
    result = summarise(measured, targets, args.rows)
    result["file"] = str(args.npz)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    verdict = "Large (suspected missing entry transition)" if result["first_max_deg"] > 10.0 else "Small (normal)"
    print(f"File: {args.npz}")
    print(f"Frames: {result['rows']}")
    print(f"First-tick max|Δ| = {result['first_max_deg']:.2f}°  RMS = {result['first_rms_deg']:.2f}°  "
          f"(worst joint #{result['first_joint']}) → {verdict}")
    print(f"Steady-state max|Δ| median = {result['settle_median_deg']:.2f}°  P90 = {result['settle_p90_deg']:.2f}°")
    print(f"Overall maximum = {result['global_max_deg']:.2f}°")
    print("Initial ticks max|Δ| (degrees): " + " ".join(f"{v:.1f}" for v in result["head_max_deg"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
