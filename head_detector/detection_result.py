from typing import List

import numpy as np

from head_detector.draw_utils import draw_3d_landmarks, draw_2d_landmarks, draw_pose, draw_bboxes
from head_detector.head_info import HeadMetadata
from head_detector.pncc_processor import PNCCProcessor


DRAW_MAPPING = {
    "landmarks": [draw_3d_landmarks],
    "points": [draw_2d_landmarks],
    "pose": [draw_pose],
    'full': [draw_bboxes, draw_3d_landmarks],
    'bbox': [draw_bboxes],
}


class PredictionResult:
    def __init__(self, original_image: np.ndarray, heads: List[HeadMetadata]):
        self.original_image = original_image
        self.heads = heads
        self.pncc_processor = PNCCProcessor()

    def draw(self, method: str = 'full'):
        image = self.original_image.copy()
        draw_methods = DRAW_MAPPING[method]
        for head in self.heads:
            for draw_method in draw_methods:
                image = draw_method(image, head)
        return image

    def get_pncc(self):
        return self.pncc_processor(self.original_image, self.heads)

    def __repr__(self):
        return f"PredictionResult(original_image={self.original_image.shape}, num heads={len(self.heads)})"
