
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .rt_config import RealtimeRetargetConfig
from booster_deploy.types.retarget_types import ReferenceFrame50HzNs, RetargetQposFrameNs


class Strict50HzRetargetPostprocessor:

    def __init__(self, config: RealtimeRetargetConfig) -> None:

        self.config = config

        self._max_emitted_idx = -1
        self.reset()

    def reset(self) -> None:

        self._emit_offset = self._max_emitted_idx + 1
        self._first_sample_ns: int | None = None
        self._next_sample_idx = 0
        self._prev_frame: RetargetQposFrameNs | None = None
        self._last_output_identity=None
        self._last_joint_pos: np.ndarray | None = None
        self._last_filtered_joint_vel: np.ndarray | None = None
        self._last_filtered_qpos: np.ndarray | None = None

    def push(self, frame: RetargetQposFrameNs) -> list[ReferenceFrame50HzNs]:

        qpos = np.asarray(frame.robot_qpos, dtype=np.float64)
        if qpos.shape[0] < 7:
            raise ValueError(f"robot_qpos must contain root pos, root quat, and joints, got {qpos.shape}")

        if self._prev_frame is not None:
            gap_ns = self._sampling_ns(frame) - self._sampling_ns(self._prev_frame)
            if gap_ns < 0 or gap_ns > int(self.config.max_retarget_gap_ns):
                self.reset()

        if self._first_sample_ns is None:
            self._first_sample_ns = self._sampling_ns(frame)
            self._prev_frame = frame
            return [self._make_reference(frame, qpos.copy(), sample_idx=0, sample_ns=self._first_sample_ns)]

        prev = self._prev_frame
        if prev is None:
            self._prev_frame = frame
            return []

        prev_ns = self._sampling_ns(prev)
        cur_ns = self._sampling_ns(frame)
        if cur_ns <= prev_ns:
            self._prev_frame = frame
            return []

        out: list[ReferenceFrame50HzNs] = []
        while True:
            sample_ns = self._sample_ns(self._next_sample_idx)
            if sample_ns > cur_ns:
                break
            if sample_ns < prev_ns:
                self._next_sample_idx += 1
                continue
            alpha = (sample_ns - prev_ns) / float(cur_ns - prev_ns)
            q = interpolate_qpos(prev.robot_qpos, frame.robot_qpos, alpha)
            out.append(self._make_reference(frame, q, sample_idx=self._next_sample_idx, sample_ns=sample_ns,weight=alpha))
        self._prev_frame = frame
        return out

    @staticmethod
    def _sampling_ns(frame):
        return int(frame.offline_sample_ns if frame.offline_sample_ns is not None else frame.body_recv_ns)

    def _sample_ns(self, sample_idx: int) -> int:

        if self._first_sample_ns is None:
            raise RuntimeError("resampler has no first sample timestamp")
        return int(self._first_sample_ns + sample_idx * self.config.sample_dt_ns)

    def _make_reference(
        self,
        source_frame: RetargetQposFrameNs,
        qpos: np.ndarray,
        *,
        sample_idx: int,
        sample_ns: int,
        weight: float | None = None,
    ) -> ReferenceFrame50HzNs:

        qpos = self._filter_qpos(qpos)
        root_pos = qpos[:3].copy()
        root_quat = _normalize_quat(qpos[3:7].copy())
        joint_pos = qpos[7:].copy()
        joint_vel = self._estimate_joint_vel(joint_pos)
        self._next_sample_idx = max(self._next_sample_idx, int(sample_idx) + 1)
        emitted_idx = int(self._emit_offset + int(sample_idx))
        self._max_emitted_idx = max(self._max_emitted_idx, emitted_idx)
        from booster_deploy.telemetry.postprocess import identity, output
        sources=tuple(dict.fromkeys((self._prev_frame.source_ids if self._prev_frame else ())+source_frame.source_ids))
        previous=getattr(self,'_last_output_identity',None)
        lineage=identity('interpolation' if weight is not None else 'resample',sources,weight=weight,source_channel='retarget_input',
            parameters={'sample_idx':emitted_idx,'sample_ns':sample_ns,'qpos_filter':self.config.qpos_filter,
                        'qpos_alpha':self.config.qpos_filter_alpha,'joint_vel_filter':self.config.joint_vel_filter,
                        'joint_vel_alpha':self.config.joint_vel_filter_alpha,'previous_output':previous})
        self._last_output_identity=lineage['id']
        output(lineage,source_ids=sources,joint_pos=joint_pos,joint_vel=joint_vel,root_pos=root_pos,root_quat=root_quat)
        return ReferenceFrame50HzNs(
            identity=lineage,
            sample_idx=emitted_idx,
            sample_ns=int(sample_ns),
            body_seq=int(source_frame.body_seq),
            source_ids=tuple(dict.fromkeys((self._prev_frame.source_ids if self._prev_frame else ()) + source_frame.source_ids)),
            clock_domain=source_frame.clock_domain,
            body_recv_ns=int(source_frame.body_recv_ns),
            retarget_done_ns=int(source_frame.retarget_done_ns),
            robot_name=str(source_frame.robot_name),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_pos=root_pos,
            root_quat=root_quat,
        )

    def _filter_qpos(self, qpos: np.ndarray) -> np.ndarray:

        if str(self.config.qpos_filter).lower() != "ema":
            return np.asarray(qpos, dtype=np.float64).copy()
        alpha = float(self.config.qpos_filter_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        qpos = np.asarray(qpos, dtype=np.float64).copy()
        if self._last_filtered_qpos is None or self._last_filtered_qpos.shape != qpos.shape:
            self._last_filtered_qpos = qpos.copy()
            return qpos
        prev = self._last_filtered_qpos
        filtered = alpha * qpos + (1.0 - alpha) * prev
        filtered[3:7] = slerp_wxyz(prev[3:7], qpos[3:7], alpha)
        self._last_filtered_qpos = filtered.copy()
        return filtered

    def _estimate_joint_vel(self, joint_pos: np.ndarray) -> np.ndarray:

        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        if self._last_joint_pos is None or self._last_joint_pos.shape != joint_pos.shape:
            self._last_joint_pos = joint_pos.copy()
            vel = np.zeros_like(joint_pos)
        else:
            vel = (joint_pos - self._last_joint_pos) / (self.config.sample_dt_ns / 1e9)
            self._last_joint_pos = joint_pos.copy()

        if self.config.max_joint_speed is not None:
            limit = abs(float(self.config.max_joint_speed))
            vel = np.clip(vel, -limit, limit)

        if str(self.config.joint_vel_filter).lower() == "ema":
            alpha = min(max(float(self.config.joint_vel_filter_alpha), 0.0), 1.0)
            if self._last_filtered_joint_vel is None or self._last_filtered_joint_vel.shape != vel.shape:
                filtered = vel
            else:
                filtered = alpha * vel + (1.0 - alpha) * self._last_filtered_joint_vel
            self._last_filtered_joint_vel = filtered.copy()
            return filtered
        return vel


def interpolate_qpos(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:

    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if q0.shape != q1.shape:
        raise ValueError(f"qpos shape mismatch: {q0.shape} vs {q1.shape}")
    alpha = min(max(float(alpha), 0.0), 1.0)
    out = (1.0 - alpha) * q0 + alpha * q1
    out[3:7] = slerp_wxyz(q0[3:7], q1[3:7], alpha)
    return out


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:

    q0 = _normalize_quat(np.asarray(q0, dtype=np.float64))
    q1 = _normalize_quat(np.asarray(q1, dtype=np.float64))
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    rots = R.from_quat(
        np.stack(
            [
                [q0[1], q0[2], q0[3], q0[0]],
                [q1[1], q1[2], q1[3], q1[0]],
            ],
            axis=0,
        )
    )
    quat_xyzw = Slerp([0.0, 1.0], rots)([min(max(float(alpha), 0.0), 1.0)]).as_quat()[0]
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _normalize_quat(q: np.ndarray) -> np.ndarray:

    q = np.asarray(q, dtype=np.float64).copy()
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm
