import os

import tqdm
import glob
import cv2
from fire import Fire
import torch
import numpy as np

from dad_3d_heads.model_training.head_mesh import HeadMesh


# region visualization
POINT_COLOR = (255, 255, 255)
HEAD_INDICES = np.load(str("/users/okupyn/head_detector/yolo_head_training/yolo_head/flame_indices/head_indices.npy"),
                          allow_pickle=True)[()]


def draw_points(image: np.ndarray, points: np.ndarray, bbox) -> np.ndarray:
    """
    Points are expected to have integer coordinates.
    """
    radius = max(1, int(min(image.shape[:2]) * 0.002))
    for pt in points:
        cv2.circle(image, (int(pt[0]) + bbox[0], int(pt[1]) + bbox[1]), radius, POINT_COLOR, -1)
    return image


def draw_3d_landmarks(projected_vertices, image: np.ndarray, bbox, triangles) -> np.ndarray:
    points = []
    points.extend(np.take(projected_vertices, np.array(HEAD_INDICES), axis=0))
    for triangle in triangles:
        pts = [(projected_vertices[i][0] + bbox[0], projected_vertices[i][1] + bbox[1]) for i in triangle]
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
    return draw_points(image, points, bbox)


def viz_folder(folder: str, output_folder: str = None):
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(folder, "*.jpg"))
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

    for file in tqdm.tqdm(files):
        if not os.path.exists(file.replace(".jpg", ".npz").replace("images", "annotations")):
            continue
        data = np.load(file.replace(".jpg", ".npz").replace("images", "annotations"))
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for index in range(len(data["3dmm_params"])):
            x, y, w, h, = data["extended_bbox"][index]
            params = data["3dmm_params"][index]
            projected_vertices = \
            head_mesh.reprojected_vertices(params_3dmm=torch.from_numpy(params), to_2d=True).numpy()[0]
            image = draw_3d_landmarks(projected_vertices, image, [x, y, w, h], filtered_triangles)
        cv2.imwrite(os.path.join(output_folder, os.path.basename(file)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    Fire(viz_folder)
