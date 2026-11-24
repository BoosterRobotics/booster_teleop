
from __future__ import annotations

from booster_deploy.hand_retargeting.somehand_core.domain.config import RetargetingConfig
from booster_deploy.hand_retargeting.somehand_core.domain.models import HandFrame, RetargetingStepResult
from booster_deploy.hand_retargeting.somehand_core.domain.preprocessing import preprocess_landmarks
from booster_deploy.hand_retargeting.somehand_core.infrastructure.config_loader import load_retargeting_config
from booster_deploy.hand_retargeting.somehand_core.infrastructure.hand_model import HandModel
from booster_deploy.hand_retargeting.somehand_core.infrastructure.vector_solver import VectorRetargeter


class RetargetingEngine:

    def __init__(self, config: RetargetingConfig, *, input_type: str = "landmarks"):
        self.config = config
        self.input_type = input_type
        self.hand_model = HandModel(config.hand.mjcf_path)
        self.retargeter = VectorRetargeter(self.hand_model, config)

    @classmethod
    def from_config_path(cls, config_path: str, *, input_type: str = "landmarks") -> "RetargetingEngine":
        return cls(load_retargeting_config(config_path), input_type=input_type)

    def describe(self) -> dict[str, int | str]:
        return {
            "model_name": self.config.hand.name,
            "dof": self.hand_model.nq,
            "vector_pairs": len(self.config.human_vector_pairs),
        }

    def process(self, frame: HandFrame) -> RetargetingStepResult:
        landmarks = frame.landmarks_3d
        if frame.hand_side != self.config.hand.side:
            raise ValueError(
                f"input hand side {frame.hand_side!r} does not match config hand side {self.config.hand.side!r}"
            )
        self.retargeter.update_targets(
            landmarks,
            hand_side=frame.hand_side,
        )
        qpos = self.retargeter.solve()
        processed_landmarks = preprocess_landmarks(
            landmarks,
            hand_side=frame.hand_side,
        )
        frame_primary_targets, frame_secondary_targets = self.retargeter.get_frame_target_directions()
        return RetargetingStepResult(
            qpos=qpos.copy(),
            target_directions=self.retargeter.get_target_directions(),
            processed_landmarks=processed_landmarks,
            hand_side=frame.hand_side,
            target_frame_primary_directions=frame_primary_targets,
            target_frame_secondary_directions=frame_secondary_targets,
            target_distances=self.retargeter.get_target_distances(),
            target_angles=self.retargeter.get_target_angles(),
        )
