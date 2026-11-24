"""Read SDK poses (XYZ + XYZW) for offline human inspection, without retargeting."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coordinate import BODY_JOINT_NAMES, body_poses_to_human_frame
from booster_deploy.types import HumanFrame


@dataclass(frozen=True)
class PicoRawRecording:
    path: Path
    poses: np.ndarray
    time_s: np.ndarray
    body_seq: np.ndarray
    fps: float
    time_source: str

    def frame_at(self, index: int, ground_shift: float = 0.0) -> HumanFrame:
        # Reuse exactly the deployed coordinate transform: (x,y,z)->(x,-z,y).
        frame = body_poses_to_human_frame(self.poses[index])
        return {
            name: (pos + [0.0, 0.0, ground_shift], quat)
            for name, (pos, quat) in frame.items()
        }


def load_pico_raw_recording(path: str | Path) -> PicoRawRecording:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        if not {'poses', 'joint_names', 'fps'}.issubset(data.files):
            raise ValueError('Expected raw Pico poses[T,24,7], joint_names and fps; robot FK NPZ is not raw Pico.')
        names = list(data['joint_names'].astype(str))
        if len(names) != 24 or set(names) != set(BODY_JOINT_NAMES):
            raise ValueError('Expected the 24 named Pico/XRobot body joints')
        poses = np.asarray(data['poses'], dtype=float)
        if poses.ndim != 3 or poses.shape[1] != 24 or poses.shape[2] < 7 or len(poses) == 0:
            raise ValueError(f'Invalid raw poses shape: {poses.shape}')
        poses = poses[:, [names.index(n) for n in BODY_JOINT_NAMES], :7].copy()
        if (
            'raw_quaternion_convention' in data
            and str(data['raw_quaternion_convention'].item()).lower() != 'xyzw'
        ):
            raise ValueError('Raw quaternion convention must be xyzw')
        if (
            'raw_pose_columns' in data
            and list(data['raw_pose_columns'][:7])
            != ['px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw']
        ):
            raise ValueError('Unexpected raw pose column order')
        fps = float(np.asarray(data['fps']).reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('FPS must be finite and positive')
        seq = np.asarray(data['body_seq'], dtype=np.int64) if 'body_seq' in data else np.arange(len(poses))
        if seq.shape != (len(poses),):
            raise ValueError('body_seq length differs from poses')
        if 'body_recv_ns' in data:
            ts = np.asarray(data['body_recv_ns'], dtype=np.int64)
            if ts.shape != (len(poses),) or np.any(np.diff(ts) <= 0):
                raise ValueError('body_recv_ns must be one strictly increasing timestamp per frame')
        else:
            ts = None
        source_ts = np.asarray(data['source_ns'], dtype=np.int64) if 'source_ns' in data else None
        if (
            source_ts is not None
            and source_ts.shape == (len(poses),)
            and np.all(source_ts > 0)
            and np.all(np.diff(source_ts) > 0)
        ):
            time_s = (source_ts - source_ts[0]).astype(float) / 1e9
            time_source = 'source_ns'
        elif ts is not None:
            time_s = (ts - ts[0]).astype(float) / 1e9
            time_source = 'body_recv_ns'
        else:
            time_s = np.arange(len(poses)) / fps
            time_source = 'fps (timestamps unavailable)'
    if not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[..., 3:7], axis=-1) < 1e-8):
        raise ValueError('Raw poses contain nonfinite values or zero quaternions')
    # Retain original quaternion values for inspection; normalize only on display.
    return PicoRawRecording(path, poses, time_s, seq, fps, time_source)


__all__ = ["PicoRawRecording", "load_pico_raw_recording"]
