"""Bounded, read-only human preview using the offline input contracts."""
from pathlib import Path
import re
import zipfile

import numpy as np

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_SOURCE_FRAMES = 12000
MAX_PREVIEW_FRAMES = 3000


def load_preview(path: Path, *, max_frames: int = MAX_PREVIEW_FRAMES) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Recording does not exist or exceeds the preview limit (32 MiB)')
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > MAX_EXPANDED_BYTES:
                raise ValueError('Uncompressed recording size exceeds the preview limit (128 MiB)')
        from booster_deploy.inputs.pico.offline_recording import load_pico_raw_recording
        from booster_deploy.inputs.pico.coordinate import pico_skeleton_connections
        clip = load_pico_raw_recording(path)
        count = len(clip.poses)
        times = clip.time_s
        time_source = clip.time_source
        frame_at = clip.frame_at
        connections = pico_skeleton_connections()
        format_name = 'raw PICO NPZ'
    else:
        # Reject huge frame declarations before the existing BVH reader allocates.
        header = path.read_text(encoding='utf-8')
        bone_count = len(re.findall(r'^\s*(?:ROOT|JOINT)\s+', header, re.M))
        if bone_count > 128:
            raise ValueError('BVH bone count exceeds the preview limit (128)')
        match = re.search(r'^Frames:\s*(\d+)\s*$', header, re.M)
        if not match or not 0 < int(match[1]) <= MAX_SOURCE_FRAMES:
            raise ValueError('Unsupported BVH or frame count exceeds the preview limit (12000)')
        del header
        from booster_deploy.inputs.mocap.offline_publisher import load_recorded_mocap_bvh, raw_joints_at
        from booster_deploy.inputs.mocap.adapter import PNLinkMocapAdapter, skeleton_connections
        clip = load_recorded_mocap_bvh(path)
        count = clip.frame_count
        times = np.arange(count) * clip.frame_time_s
        time_source = 'BVH Frame Time'
        adapter = PNLinkMocapAdapter()
        frame_at = lambda index: adapter.joints_to_human_frame(raw_joints_at(clip, index))
        connections = skeleton_connections()
        format_name = 'PN-Link BVH'
    if count > MAX_SOURCE_FRAMES:
        raise ValueError('Recording frame count exceeds the preview limit (12000)')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Recording timeline is invalid')
    indices = np.unique(np.linspace(0, count - 1, min(count, max(1, min(int(max_frames), MAX_PREVIEW_FRAMES))), dtype=int))
    first = frame_at(int(indices[0]))
    names = sorted(first)
    lookup = {name: i for i, name in enumerate(names)}
    rows = []
    for i in indices:
        frame = frame_at(int(i))
        rows.append([frame[n][0] for n in names])
    positions = np.asarray(rows)
    if not np.isfinite(positions).all():
        raise ValueError('Recording skeleton contains non-finite coordinates')
    return dict(kind='human', filename=path.name, format=format_name, names=names,
                edges=[[lookup[a], lookup[b]] for a, b in connections if a in lookup and b in lookup],
                positions=positions.reshape(len(indices), -1).round(5).tolist(),
                time_s=times[indices].tolist(), time_source=time_source, source_frames=count)
