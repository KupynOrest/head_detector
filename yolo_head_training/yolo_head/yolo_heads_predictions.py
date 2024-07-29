import dataclasses
from typing import Union

import numpy as np
from torch import Tensor


@dataclasses.dataclass
class YoloHeadsPredictions:
    """
    A data class that encapsulates pose estimation predictions for a single image.

    :param scores:       Array of shape [N] with scores for each pose with [0..1] range.
    :param bboxes_xyxy:  Array of shape [N, 4] with bounding boxes for each pose in XYXY format.
    """

    scores: Union[Tensor, np.ndarray]
    bboxes_xyxy: Union[Tensor, np.ndarray]
    mm_params: Union[Tensor, np.ndarray]
    predicted_3d_vertices: Union[Tensor, np.ndarray]
    predicted_2d_vertices: Union[Tensor, np.ndarray]