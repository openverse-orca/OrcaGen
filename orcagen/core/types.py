from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Pose:
    pos: np.ndarray  # shape (3,)
    mat: np.ndarray  # shape (3,3)
    quat: np.ndarray  # shape (4,) [w,x,y,z]

