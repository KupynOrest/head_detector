import json
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np

# Pair of indexes corresponding to the 68 face keypoints
# Indexes are 1-based
FACE_KEYPOINTS_SKELETON = (
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    #
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
    #
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    #
    (28, 29),
    (29, 30),
    (30, 31),
    #
    (32, 33),
    (33, 34),
    (34, 35),
    (35, 36),
    #
    (37, 38),
    (38, 39),
    (39, 40),
    (40, 41),
    (41, 42),
    (42, 37),
    #
    (43, 44),
    (44, 45),
    (45, 46),
    (46, 47),
    (47, 48),
    (48, 43),
    #
    (49, 50),
    (50, 51),
    (51, 52),
    (52, 53),
    (53, 54),
    (54, 55),
    (55, 56),
    (56, 57),
    (57, 58),
    (58, 59),
    (59, 60),
    (60, 49),
    #
    (61, 62),
    (62, 63),
    (63, 64),
    (64, 65),
    (65, 66),
    (66, 67),
    (67, 68),
    (68, 61),
)

FACE_KEYPOINTS_FLIP_INDEXES = (
    (1, 17),
    (2, 16),
    (3, 15),
    (4, 14),
    (5, 13),
    (6, 12),
    (7, 11),
    (8, 10),
    (18, 27),
    (19, 26),
    (20, 25),
    (21, 24),
    (22, 23),
    (37, 46),
    (38, 45),
    (39, 44),
    (40, 43),
    (42, 47),
    (41, 48),
    (32, 36),
    (33, 35),
    #
    (49, 55),
    (50, 54),
    (61, 65),
    (60, 56),
    #
    (51, 53),
    (62, 64),
    (68, 66),
    (59, 57),
)


@dataclass
class HeadAnnotation:
    """
    :param bbox: (x,y,w,h)
    :param extended_bbox: (x,y,w,h)
    :param points: (68, 2) - Face keypoints relative to bbox
    """

    bbox: Tuple[int, int, int, int]
    extended_bbox: Tuple[int, int, int, int]
    points: np.ndarray
    projected_vertices: np.ndarray
    vertices_3d: np.ndarray
    mm_params: np.ndarray

    def get_face_bbox_xywh(self):
        return np.array(self.bbox)

    def get_face_bbox_xyxy(self):
        return np.array([self.bbox[0], self.bbox[1], self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]])

    def get_extended_face_bbox_xywh(self):
        return np.array(self.extended_bbox)

    def get_extended_face_bbox_xyxy(self):
        return np.array(
            [
                self.extended_bbox[0],
                self.extended_bbox[1],
                self.extended_bbox[0] + self.extended_bbox[2],
                self.extended_bbox[1] + self.extended_bbox[3],
            ]
        )

    def get_points_in_absolute_coords(self):
        return self.points + np.array(self.extended_bbox[:2])

    def get_reprojected_points_in_absolute_coords(self):
        return self.projected_vertices + np.array(self.extended_bbox[:2])


@dataclass
class SampleAnnotation:
    heads: List[HeadAnnotation]


def read_annotation(ann_file: str) -> SampleAnnotation:
    with open(ann_file, "r") as f:
        ann = json.load(f)

    head_anns = []
    for head in ann:
        head_anns.append(
            HeadAnnotation(
                bbox=tuple(head["bbox"]),
                extended_bbox=tuple(head["extended_bbox"]),
                points=np.array(head["points"], dtype=np.float32),
                projected_vertices=np.array(head["projected_vertices"], dtype=np.float32),
                vertices_3d=np.array(head["3d_vertices"], dtype=np.float32),
                mm_params=np.array(head["3dmm_params"], dtype=np.float32),
            )
        )
    sample = SampleAnnotation(heads=head_anns)
    return sample


def draw_2d_keypoints(image, keypoints, color=(0, 0, 255), radius=4, opacity=0.5):
    overlay = image.copy()
    for kp in keypoints:
        cv2.circle(overlay, (int(kp[0]), int(kp[1])), radius, color, -1)
    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, dst=overlay)


def draw_face_keypoints_skeleton(image, keypoints, color=(0, 0, 255), thickness=4, opacity=0.5):
    overlay = image.copy()
    if len(keypoints) != 68:
        raise ValueError("Expected 68 keypoints, got %d" % len(keypoints))
    for i, j in FACE_KEYPOINTS_SKELETON:
        cv2.line(
            overlay,
            (int(keypoints[i - 1][0]), int(keypoints[i - 1][1])),
            (int(keypoints[j - 1][0]), int(keypoints[j - 1][1])),
            color,
            thickness,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, dst=overlay)


__all__ = (
    "read_annotation",
    "SampleAnnotation",
    "HeadAnnotation",
    "FACE_KEYPOINTS_SKELETON",
    "FACE_KEYPOINTS_FLIP_INDEXES",
    "draw_face_keypoints_skeleton",
    "draw_2d_keypoints",
)

if __name__ == "__main__":
    read_annotation("sample.json")
