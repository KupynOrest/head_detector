from typing import Optional, Tuple

from math import sqrt, cos, sin

import cv2
import numpy as np

from head_detector.head_info import HeadMetadata
from head_detector.utils import get_relative_path


FACE_INDICES = np.load(str(get_relative_path("assets/flame_indices/face.npy", __file__)),
                          allow_pickle=True)[()]
HEAD_INDICES = np.load(str(get_relative_path("assets/flame_indices/head_indices.npy", __file__)),
                          allow_pickle=True)[()]
TRIANGLES = np.loadtxt(get_relative_path("assets/triangles.txt", __file__), delimiter=',').astype(np.int32)
POINT_COLOR = (255, 255, 255)


def draw_points(image: np.ndarray, points: np.ndarray, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Points are expected to have integer coordinates.
    """
    if color is None:
        color = POINT_COLOR
    radius = max(1, int(min(image.shape[:2]) * 0.001))
    for pt in points:
        cv2.circle(image, (int(pt[0]), int(pt[1])), radius, color, -1)
    return image


def draw_2d_landmarks(image: np.ndarray, head: HeadMetadata) -> np.ndarray:
    points = []
    points.extend(np.take(head.vertices_3d[:, :2], np.array(FACE_INDICES), axis=0))
    return draw_points(image, np.array(points))


def draw_3d_landmarks(image: np.ndarray, head: HeadMetadata) -> np.ndarray:
    points = []
    projected_vertices = head.vertices_3d[:, :2]
    points.extend(np.take(projected_vertices, np.array(HEAD_INDICES), axis=0))
    for triangle in TRIANGLES:
        pts = [(projected_vertices[i][0], projected_vertices[i][1]) for i in triangle]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
    return draw_points(image, np.array(points))


def draw_pose(image: np.ndarray, head: HeadMetadata) -> np.ndarray:
    rpy = head.head_pose
    roll, pitch, yaw = rpy.roll, rpy.pitch, rpy.yaw
    bbox = head.bbox
    bbox_area = bbox.w * bbox.h
    tdx, tdy = bbox.x + bbox.w // 2, bbox.y + bbox.h // 2

    size = sqrt(bbox_area) // 4

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = image.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    thickness = max(1, int(sqrt(bbox_area) * 0.03))
    cv2.arrowedLine(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), thickness)
    cv2.arrowedLine(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), thickness)
    cv2.arrowedLine(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), thickness)

    return image


def draw_bboxes(image: np.ndarray, head: HeadMetadata) -> np.ndarray:
    x, y, w, h = head.bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image
