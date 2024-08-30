import os
from typing import Union, Tuple

import cv2
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from head_detector.head_info import RPY, FlameParams, Bbox


def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)


IMAGE_SIZE = 640
FACE_INDICES = np.load(str(get_relative_path("assets/flame_indices/face.npy", __file__)),
                          allow_pickle=True)[()]
HEAD_INDICES = np.load(str(get_relative_path("assets/flame_indices/head_indices.npy", __file__)),
                          allow_pickle=True)[()]
TRIANGLES = np.loadtxt(get_relative_path("assets/triangles.txt", __file__), delimiter=',').astype(np.int32)


def refined_head_bbox(vertices: np.ndarray) -> Bbox:
    points = []
    points.extend(np.take(vertices, np.array(HEAD_INDICES), axis=0))
    points = np.array(points)
    x = min(points[:, 0])
    y = min(points[:, 1])
    x1 = max(points[:, 0])
    y1 = max(points[:, 1])
    x, y, x1, y1 = list(map(int, [x, y, x1, y1]))
    return Bbox(x=x, y=y, w=x1 - x, h=y1 - y)


def extend_bbox(bbox: np.array, offset: Union[Tuple[float, ...], float] = 0.1) -> np.array:
    """
    Increases bbox dimensions by offset*100 percent on each side.

    IMPORTANT: Should be used with ensure_bbox_boundaries, as might return negative coordinates for x_new, y_new,
    as well as w_new, h_new that are greater than the image size the bbox is extracted from.

    :param bbox: [x, y, w, h]
    :param offset: (left, right, top, bottom), or (width_offset, height_offset), or just single offset that specifies
    fraction of spatial dimensions of bbox it is increased by.

    For example, if bbox is a square 100x100 pixels, and offset is 0.1, it means that the bbox will be increased by
    0.1*100 = 10 pixels on each side, yielding 120x120 bbox.

    :return: extended bbox, [x_new, y_new, w_new, h_new]
    """
    x, y, w, h = bbox

    if isinstance(offset, tuple):
        if len(offset) == 4:
            left, right, top, bottom = offset
        elif len(offset) == 2:
            w_offset, h_offset = offset
            left = right = w_offset
            top = bottom = h_offset
    else:
        left = right = top = bottom = offset

    return np.array([x - w * left, y - h * top, w * (1.0 + right + left), h * (1.0 + top + bottom)]).astype("int32")


def extend_to_rect(bbox: np.array) -> np.array:
    x, y, w, h = bbox
    if w > h:
        diff = w - h
        return np.array([x, y - diff // 2, w, w])
    else:
        diff = h - w
        return np.array([x - diff // 2, y, h, h])


def flame_params_skull_center(flame_params: FlameParams, image: np.ndarray) -> Tuple[int, int]:
    h, w = image.shape[:2]
    scale = IMAGE_SIZE / max(image.shape[:2])
    if h > w:
        new_h, new_w = IMAGE_SIZE, int(w * IMAGE_SIZE / h)
    else:
        new_h, new_w = int(h * IMAGE_SIZE / w), IMAGE_SIZE
    pad_w = IMAGE_SIZE - new_w
    pad_h = IMAGE_SIZE - new_h
    scull_center = flame_params.translation / scale
    scull_center = scull_center[0].numpy()
    return int(scull_center[0] - pad_w), int(scull_center[1] - pad_h)


def get_rotation_mat(
    img: np.ndarray, img_center: Tuple[int, int], angle: Union[float, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = img.shape[:2]
    rotation_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - img_center[0]
    rotation_mat[1, 2] += bound_h / 2 - img_center[1]
    return rotation_mat, (bound_w, bound_h)


def vertically_align(
    img: np.ndarray, vertices: np.ndarray, flame_params, roll: float
):
    scull_center = flame_params_skull_center(flame_params, img)
    rot_mat, bounds = get_rotation_mat(img, scull_center, roll)
    vertical_img = cv2.warpAffine(img, rot_mat, bounds, flags=cv2.INTER_LINEAR)
    vertices = np.hstack([vertices[:, :2], np.ones((vertices.shape[0], 1))])
    rotated_landmarks = vertices @ rot_mat.T
    return vertical_img, rotated_landmarks


def rot_mat_from_6dof(v: torch.Tensor) -> torch.Tensor:
    assert v.shape[-1] == 6
    v = v.view(-1, 6)
    vx, vy = v[..., :3].clone(), v[..., 3:].clone()

    b1 = F.normalize(vx, dim=-1)
    b3 = F.normalize(torch.cross(b1, vy, dim=-1), dim=-1)
    b2 = -torch.cross(b1, b3, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def limit_angle(angle: Union[int, float], pi: Union[int, float] = 180.0) -> Union[int, float]:
    """
    Angle should be in degrees, not in radians.
    If you have an angle in radians - use the function radians_to_degrees.
    """
    if angle < -pi:
        k = -2 * (int(angle / pi) // 2)
        angle = angle + k * pi
    if angle > pi:
        k = 2 * ((int(angle / pi) + 1) // 2)
        angle = angle - k * pi

    return angle


def calculate_rpy(flame_params) -> RPY:
    rot_mat = rotation_mat_from_flame_params(flame_params)
    rot_mat_2 = np.transpose(rot_mat)
    angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
    roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
    return RPY(roll=roll, pitch=pitch, yaw=yaw)


def rotation_mat_from_flame_params(flame_params):
    rot_mat = rot_mat_from_6dof(flame_params.rotation).numpy()[0]
    return rot_mat


def nms(
        boxes_xyxy,
        scores,
        flame_params,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        top_k: int = 1000,
        keep_top_k: int = 100
):
    for pred_bboxes_xyxy, pred_bboxes_conf, pred_flame_params in zip(
            boxes_xyxy.detach().float(),
            scores.detach().float(),
            flame_params.detach().float(),
    ):
        pred_bboxes_conf = pred_bboxes_conf.squeeze(-1)  # [Anchors]
        conf_mask = pred_bboxes_conf >= confidence_threshold

        pred_bboxes_conf = pred_bboxes_conf[conf_mask]
        pred_bboxes_xyxy = pred_bboxes_xyxy[conf_mask]
        pred_flame_params = pred_flame_params[conf_mask]

        # Filter all predictions by self.nms_top_k
        if pred_bboxes_conf.size(0) > top_k:
            topk_candidates = torch.topk(pred_bboxes_conf, k=top_k, largest=True, sorted=True)
            pred_bboxes_conf = pred_bboxes_conf[topk_candidates.indices]
            pred_bboxes_xyxy = pred_bboxes_xyxy[topk_candidates.indices]
            pred_flame_params = pred_flame_params[topk_candidates.indices]

        # NMS
        idx_to_keep = torchvision.ops.boxes.nms(boxes=pred_bboxes_xyxy, scores=pred_bboxes_conf,
                                                iou_threshold=iou_threshold)

        final_bboxes = pred_bboxes_xyxy[idx_to_keep][: keep_top_k]  # [Instances, 4]
        final_scores = pred_bboxes_conf[idx_to_keep][: keep_top_k]  # [Instances, 1]
        final_params = pred_flame_params[idx_to_keep][: keep_top_k]  # [Instances, Flame Params]
        return final_bboxes, final_scores, final_params