import os
from typing import List
from dataclasses import dataclass
from collections import namedtuple

import cv2
import numpy as np


def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)


Bbox = namedtuple("Bbox", ["x", "y", "w", "h"])
RPY = namedtuple("RPY", ["roll", "pitch", "yaw"])
FACE_INDICES = np.load(str(get_relative_path("../assets/flame_indices/face.npy", __file__)),
                          allow_pickle=True)[()]
HEAD_INDICES = np.load(str(get_relative_path("../assets/flame_indices/head_indices.npy", __file__)),
                          allow_pickle=True)[()]
TRIANGLES = np.loadtxt(get_relative_path("../assets/triangles.txt", __file__), delimiter=',').astype(np.int32)
POINT_COLOR = (255, 255, 255)


def draw_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Points are expected to have integer coordinates.
    """
    radius = max(1, int(min(image.shape[:2]) * 0.001))
    for pt in points:
        cv2.circle(image, (int(pt[0]), int(pt[1])), radius, POINT_COLOR, -1)
    return image


def draw_3d_landmarks(projected_vertices, image: np.ndarray) -> np.ndarray:
    points = []
    points.extend(np.take(projected_vertices, np.array(HEAD_INDICES), axis=0))
    for triangle in TRIANGLES:
        pts = [(projected_vertices[i][0], projected_vertices[i][1]) for i in triangle]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
    return draw_points(image, np.array(points))


@dataclass
class HeadMetadata:
    bbox: Bbox
    score: float
    flame_params: np.ndarray
    vertices_3d: np.ndarray
    head_pose: RPY


class PredictionResult:
    def __init__(self, original_image: np.ndarray, heads: List[HeadMetadata]):
        self.original_image = original_image
        self.heads = heads

    def draw(self):
        image = self.original_image.copy()
        for head in self.heads:
            x, y, w, h = head.bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            image = draw_3d_landmarks(head.vertices_3d, image)
        return image
