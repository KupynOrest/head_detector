from dataclasses import dataclass
from collections import namedtuple

import numpy as np

Bbox = namedtuple("Bbox", ["x", "y", "w", "h"])
RPY = namedtuple("RPY", ["roll", "pitch", "yaw"])


@dataclass
class HeadMetadata:
    bbox: Bbox
    score: float
    flame_params: np.ndarray
    vertices_3d: np.ndarray
    head_pose: RPY
