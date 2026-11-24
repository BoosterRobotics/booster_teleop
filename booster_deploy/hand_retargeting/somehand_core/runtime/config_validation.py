
from __future__ import annotations

from pathlib import Path

from booster_deploy.hand_retargeting.somehand_core.domain.config import RetargetingConfig
from booster_deploy.hand_retargeting.somehand_core.external_assets import build_missing_asset_message


def validate_runtime_retargeting_config(config: RetargetingConfig) -> None:
    if not Path(config.hand.mjcf_path).exists():
        raise FileNotFoundError(
            build_missing_asset_message(
                config.hand.mjcf_path,
                label="MJCF file",
            )
        )

