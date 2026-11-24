
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from booster_deploy.utils.diag_report import reporter
from booster_deploy.utils.input_health import InputHealth, SourceIdentity
from booster_deploy.types import HumanFrame
from booster_deploy.types.retarget_types import RetargetInputFrameProtocol
from booster_deploy.retargeting.gmr.params import IK_CONFIG_DICT, ROBOT_XML_DICT
from booster_deploy.retargeting.gmr_adapter import GMRRetargeter

from .human_height import HumanHeightResolution, resolve_actual_human_height
from .rt_config import RealtimeRetargetConfig
from .rt_resampler import Strict50HzRetargetPostprocessor
from booster_deploy.types.retarget_types import ReferenceFrame50HzNs, RetargetQposFrameNs


class RealtimeRetargetWorker:

    def __init__(self, config: RealtimeRetargetConfig) -> None:

        self.config = config
        self._diag = reporter("retarget_backend")
        self._health = InputHealth()
        self._source_identity = SourceIdentity()
        self._retargeter: GMRRetargeter | None = None
        self._human_format: str | None = None
        self._human_height: HumanHeightResolution | None = None
        self._postprocessor = Strict50HzRetargetPostprocessor(config)

    @property
    def postprocessor(self) -> Strict50HzRetargetPostprocessor:

        return self._postprocessor

    @property
    def last_scaled_human_frame(self) -> HumanFrame | None:
        """Expose the latest display targets without running a second IK solve."""

        if self._retargeter is None:
            return None
        return self._retargeter.last_scaled_human_frame

    def retarget_one(self, frame: RetargetInputFrameProtocol) -> RetargetQposFrameNs:

        offline_ns = getattr(frame, 'offline_sample_ns', None)
        if offline_ns is not None:
            epoch = (getattr(frame, 'stream_id', None), getattr(frame, 'loop', None),getattr(frame,'offline_cycle',0))
            if epoch != getattr(self, '_offline_epoch', None):
                self.reset()
                self._offline_epoch = epoch
        if self._retargeter is None:
            self._human_format = str(frame.human_format)
            if self.config.human_format is not None and self.config.human_format != self._human_format:
                raise ValueError(
                    f"Input human_format={self._human_format!r} does not match config.human_format={self.config.human_format!r}"
                )
            self._human_height = resolve_actual_human_height(
                human_format=self._human_format,
                robot_name=self.config.robot_name,
                frame=frame.human_frame,
                configured_height=self.config.actual_human_height,
            )
            self._print_config_match_banner(self._human_format, self.config.robot_name, self._human_height)
            self._retargeter = GMRRetargeter(
                human_format=self._human_format,
                robot_name=self.config.robot_name,
                actual_human_height=self._human_height.value_m,
                solver=self.config.solver,
                ground_detection_mode=self.config.ground_detection_mode,
                use_velocity_limit=self.config.use_velocity_limit,
                source=self.config.source,
            )
        elif self._human_format != str(frame.human_format):
            raise ValueError(f"human_format changed from {self._human_format!r} to {frame.human_format!r}")

        retarget_recv_ns = time.monotonic_ns()
        source_ids = self._source_identity.ids(frame)
        progress = self._health.observe(source_ids, retarget_recv_ns)
        retarget_frame = self._retargeter.retarget(
            frame.human_frame,
            timestamp_s=(getattr(frame, "offline_sample_ns", None) or frame.recv_ns) / 1e9,
            seq=frame.seq,
        )
        retarget_done_ns = time.monotonic_ns()
        from booster_deploy.telemetry.retarget import consumed
        consumed(frame, retarget_frame.qpos, retarget_recv_ns, retarget_done_ns, source_ids)
        self._diag.note(input_health=self._health.snapshot(retarget_done_ns))
        self._diag.mark("loop", retarget_done_ns)
        if progress == "new": self._diag.count("new_input", retarget_done_ns)
        self._diag.emit("retarget backend")
        return RetargetQposFrameNs(
            body_seq=int(frame.seq),
            source_ids=source_ids,
            clock_domain=getattr(frame, "clock_domain", None),
            body_recv_ns=int(frame.recv_ns),
            offline_sample_ns=getattr(frame,"offline_sample_ns",None),offline_cycle=getattr(frame,"offline_cycle",0),
            retarget_recv_ns=retarget_recv_ns,
            retarget_done_ns=retarget_done_ns,
            robot_name=self.config.robot_name,
            robot_qpos=np.asarray(retarget_frame.qpos, dtype=np.float64),
        )

    def postprocess_one(self, frame: RetargetQposFrameNs) -> list[ReferenceFrame50HzNs]:

        return [replace(ref,offline_cycle=frame.offline_cycle) for ref in self._postprocessor.push(frame)]

    def process_one(self, frame: RetargetInputFrameProtocol) -> list[ReferenceFrame50HzNs]:

        return self.postprocess_one(self.retarget_one(frame))

    def reset(self) -> None:

        self._retargeter = None
        self._human_format = None
        self._human_height = None
        self._postprocessor.reset()

    def close(self) -> None:

        self._diag.close()
        self.reset()

    @staticmethod
    def _print_config_match_banner(
        human_format: str, robot_name: str, human_height: HumanHeightResolution,
    ) -> None:

        ik_path = IK_CONFIG_DICT.get(human_format, {}).get(robot_name)
        xml_path = ROBOT_XML_DICT.get(robot_name)
        line = "=" * 96
        bold_cyan = "\033[1;96m"
        bold_green = "\033[1;92m"
        reset = "\033[0m"
        print(f"{bold_cyan}{line}{reset}", flush=True)
        print(f"{bold_green}REALTIME RETARGET CONFIG MATCH{reset}", flush=True)
        print(f"{bold_cyan}source human_format : {human_format}{reset}", flush=True)
        print(f"{bold_cyan}target robot        : {robot_name}{reset}", flush=True)
        print(f"{bold_cyan}ik json             : {ik_path}{reset}", flush=True)
        print(f"{bold_cyan}robot xml           : {xml_path}{reset}", flush=True)
        print(
            f"{bold_cyan}actual human height : {human_height.value_m:.4f} m "
            f"({human_height.source}){reset}",
            flush=True,
        )
        print(f"{bold_cyan}{line}{reset}", flush=True)
