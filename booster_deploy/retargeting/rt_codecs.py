
from __future__ import annotations

from typing import Any

import numpy as np

from booster_deploy.types.retarget_types import ReferenceFrame50HzNs, RetargetQposFrameNs
from booster_deploy.hand_retargeting.codecs import decode_hand_retarget_frame, encode_hand_retarget_frame
from booster_deploy.inputs.mocap.codecs import decode_mocap_retarget_input, encode_mocap_retarget_input
from booster_deploy.types.mocap_types import MocapRetargetInputFrameNs


def encode_retarget_qpos(frame: RetargetQposFrameNs) -> bytes:

    import msgpack

    payload = {
        "body_seq": int(frame.body_seq),
        "source_ids": frame.source_ids,
        "clock_domain": frame.clock_domain,
        "body_recv_ns": int(frame.body_recv_ns),
        "retarget_recv_ns": int(frame.retarget_recv_ns),
        "retarget_done_ns": int(frame.retarget_done_ns),
        "robot_name": str(frame.robot_name),
        "robot_qpos": np.asarray(frame.robot_qpos, dtype=np.float64).tolist(),
    }
    return msgpack.packb(payload, use_bin_type=True)


def decode_retarget_qpos(payload: bytes) -> RetargetQposFrameNs:

    import msgpack

    raw = msgpack.unpackb(payload, raw=False)
    return RetargetQposFrameNs(
        body_seq=int(raw["body_seq"]),
        source_ids=tuple(tuple(x) for x in raw.get("source_ids", ())),
        clock_domain=raw.get("clock_domain"),
        body_recv_ns=int(raw["body_recv_ns"]),
        retarget_recv_ns=int(raw["retarget_recv_ns"]),
        retarget_done_ns=int(raw["retarget_done_ns"]),
        robot_name=str(raw["robot_name"]),
        robot_qpos=np.asarray(raw["robot_qpos"], dtype=np.float64),
    )


def encode_reference_50hz(frame: ReferenceFrame50HzNs) -> bytes:

    import msgpack

    payload: dict[str, Any] = {
        "sample_idx": int(frame.sample_idx),
        "offline_cycle": int(frame.offline_cycle),
        "identity": frame.identity,
        "sample_ns": int(frame.sample_ns),
        "body_seq": int(frame.body_seq),
        "source_ids": frame.source_ids,
        "clock_domain": frame.clock_domain,
        "body_recv_ns": int(frame.body_recv_ns),
        "retarget_done_ns": int(frame.retarget_done_ns),
        "robot_name": str(frame.robot_name),
        "joint_pos": np.asarray(frame.joint_pos, dtype=np.float64).tolist(),
        "joint_vel": np.asarray(frame.joint_vel, dtype=np.float64).tolist(),
        "root_pos": np.asarray(frame.root_pos, dtype=np.float64).tolist(),
        "root_quat": np.asarray(frame.root_quat, dtype=np.float64).tolist(),
    }
    if frame.human_frame is not None:
        payload["human_format"] = "" if frame.human_format is None else str(frame.human_format)
        payload["human_frame"] = encode_mocap_retarget_input(
            MocapRetargetInputFrameNs(
                body_seq=int(frame.body_seq),
                body_recv_ns=int(frame.body_recv_ns),
                convert_done_ns=0,
                human_format="" if frame.human_format is None else str(frame.human_format),
                human_frame=frame.human_frame,
            )
        )
    elif frame.human_format is not None:
        payload["human_format"] = str(frame.human_format)
    if frame.hand_frame is not None:
        payload["hand_frame"] = encode_hand_retarget_frame(frame.hand_frame)
    return msgpack.packb(payload, use_bin_type=True)


def decode_reference_50hz(payload: bytes) -> ReferenceFrame50HzNs:

    import msgpack

    raw = msgpack.unpackb(payload, raw=False)
    human_format = raw.get("human_format")
    human_frame = None
    if raw.get("human_frame") is not None:
        human_payload = raw["human_frame"]
        if isinstance(human_payload, str):
            human_payload = human_payload.encode("latin1")
        decoded_human = decode_mocap_retarget_input(human_payload)
        human_format = decoded_human.human_format
        human_frame = decoded_human.human_frame
    hand_frame = None
    if raw.get("hand_frame") is not None:
        hand_payload = raw["hand_frame"]
        if isinstance(hand_payload, str):
            hand_payload = hand_payload.encode("latin1")
        hand_frame = decode_hand_retarget_frame(hand_payload)
    return ReferenceFrame50HzNs(
        sample_idx=int(raw["sample_idx"]),
        offline_cycle=int(raw.get("offline_cycle",0)),
        identity=raw.get("identity"),
        sample_ns=int(raw["sample_ns"]),
        body_seq=int(raw["body_seq"]),
        source_ids=tuple(tuple(x) for x in raw.get("source_ids", ())),
        clock_domain=raw.get("clock_domain"),
        body_recv_ns=int(raw["body_recv_ns"]),
        retarget_done_ns=int(raw["retarget_done_ns"]),
        robot_name=str(raw["robot_name"]),
        joint_pos=np.asarray(raw["joint_pos"], dtype=np.float64),
        joint_vel=np.asarray(raw["joint_vel"], dtype=np.float64),
        root_pos=np.asarray(raw["root_pos"], dtype=np.float64),
        root_quat=np.asarray(raw["root_quat"], dtype=np.float64),
        human_format=None if human_format is None else str(human_format),
        human_frame=human_frame,
        hand_frame=hand_frame,
    )
