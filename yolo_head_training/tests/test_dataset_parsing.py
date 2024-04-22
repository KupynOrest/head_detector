import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from yolo_head.dataset_parsing import (
    read_annotation,
    draw_face_keypoints_skeleton,
    draw_2d_keypoints,
    FACE_KEYPOINTS_FLIP_INDEXES,
    FACE_KEYPOINTS_SKELETON,
)
from yolo_head.flame import get_445_keypoints_indexes

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_bbox(image, bbox: Tuple[float, float, float, float], color=(0, 0, 255), thickness=2, opacity=0.5):
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (int(bbox[0]), int(bbox[1])),
        (
            int(bbox[2]),
            int(bbox[3]),
        ),
        color,
        thickness,
    )
    return cv2.addWeighted(overlay, 1 - opacity, image, opacity, 0, dst=overlay)


def test_draw_reprojected_points():
    image = cv2.imread(os.path.join(CURRENT_DIR, "1.jpg"))
    ann = read_annotation(os.path.join(CURRENT_DIR, "1.json"))

    for head in ann.heads:
        image = draw_2d_keypoints(image, head.get_reprojected_points_in_absolute_coords())

    plt.figure(figsize=(8, 8))
    plt.imshow(image[:, :, ::-1])
    plt.tight_layout()
    plt.savefig("head_get_reprojected_points_in_absolute_coords.jpg")
    plt.show()

def test_draw_selected_points():
    image = cv2.imread(os.path.join(CURRENT_DIR, "1.jpg"))
    ann = read_annotation(os.path.join(CURRENT_DIR, "1.json"))
    indexes = get_445_keypoints_indexes()

    for head in ann.heads:
        pts = head.get_reprojected_points_in_absolute_coords()
        image = draw_2d_keypoints(image, pts[indexes])

    plt.figure(figsize=(8, 8))
    plt.imshow(image[:, :, ::-1])
    plt.tight_layout()
    plt.savefig("test_draw_selected_points.jpg")
    plt.show()


def test_draw_points():
    image = cv2.imread(os.path.join(CURRENT_DIR, "1.jpg"))
    ann = read_annotation(os.path.join(CURRENT_DIR, "1.json"))

    for head in ann.heads:
        points = head.get_68_face_landmarks_in_absolute_coords()
        image = draw_face_keypoints_skeleton(image, points)
        image = draw_2d_keypoints(image, points)
        image = draw_bbox(image, head.get_face_bbox_xyxy(), color=(0, 255, 0))
        image = draw_bbox(image, head.get_extended_face_bbox_xyxy(), color=(255, 0, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(image[:, :, ::-1])
    plt.tight_layout()
    plt.savefig("head_get_points_in_absolute_coords.jpg")
    plt.show()


def test_skeleton_indexes_to_yaml():
    pairs = np.array(FACE_KEYPOINTS_SKELETON) - 1
    for pair in pairs:
        print("- [%d, %d]" % (pair[0], pair[1]))


def test_flip_indexes_to_yaml():
    order = np.arange(68)

    for i, j in FACE_KEYPOINTS_FLIP_INDEXES:
        order[i - 1], order[j - 1] = order[j - 1], order[i - 1]

    print(order.tolist())


def test_default_oks_sigmas_to_yaml():
    print([0.025] * 68)


def test_default_colors_to_yaml():
    edge_colors = [[0, 0, 255]] * len(FACE_KEYPOINTS_SKELETON)
    keypoint_colors = [[0, 0, 255]] * 68
    print("edge_colors:", edge_colors)
    print("keypoint_colors:", keypoint_colors)
