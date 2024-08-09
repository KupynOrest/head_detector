import os
from typing import Union

import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from head_detector.head_info import RPY


def get_relative_path(x: str, rel_to: str) -> str:
    return os.path.join(os.path.dirname(rel_to), x)


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