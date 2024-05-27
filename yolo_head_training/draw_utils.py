import os
from math import cos, sin

import cv2
import numpy as np
from pytorch_toolbelt.utils import vstack_header
from head_mesh import HeadMesh

from yolo_head.flame import RPY


def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)


POINT_COLOR = (255, 255, 255)
HEAD_INDICES = np.load(str(get_relative_path("yolo_head/flame_indices/head_indices.npy", __file__)),
                          allow_pickle=True)[()]

def get_triangles():
    head_mesh = HeadMesh()
    triangles = head_mesh.flame.faces
    filtered_triangles = []
    for triangle in triangles:
        keep = True
        for v in triangle:
            if v not in HEAD_INDICES:
                keep = False
        if keep:
            filtered_triangles.append(triangle)
    return filtered_triangles

FILTERED_TRAINGLES = get_triangles()


def draw_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Points are expected to have integer coordinates.
    """
    radius = max(1, int(min(image.shape[:2]) * 0.002))
    for pt in points:
        cv2.circle(image, (int(pt[0]), int(pt[1])), 1, POINT_COLOR, -1)
    return image


def draw_3d_landmarks(projected_vertices, image: np.ndarray) -> np.ndarray:
    points = []
    points.extend(np.take(projected_vertices, np.array(HEAD_INDICES), axis=0))
    for triangle in FILTERED_TRAINGLES:
        pts = [(projected_vertices[i][0], projected_vertices[i][1]) for i in triangle]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
    return draw_points(image, points)


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
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

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def draw_pose(rpy: RPY, image: np.ndarray) -> np.ndarray:
    image = draw_axis(image, rpy.yaw, rpy.pitch, rpy.roll)
    image = vstack_header(image, f"R: {rpy.roll:.2f}, P: {rpy.pitch:.2f}, Y: {rpy.yaw:.2f}")
    return image
