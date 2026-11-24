
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray


HumanFrame = Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]


def clone_human_frame(frame: HumanFrame) -> HumanFrame:

    return {
        name: (
            np.asarray(pos, dtype=np.float64).copy(),
            np.asarray(quat, dtype=np.float64).copy(),
        )
        for name, (pos, quat) in frame.items()
    }
