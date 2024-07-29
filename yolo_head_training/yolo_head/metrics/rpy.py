import math
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn
from scipy.spatial.transform import Rotation
from super_gradients.common.registry import register_metric
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy
from torchmetrics import Metric

from .functional import match_head_boxes
from ..flame import FlameParams, FLAME_CONSTS, RPY, rot_mat_from_6dof
from ..mesh_sample import MeshEstimationSample
from ..yolo_heads_post_prediction_callback import YoloHeadsPostPredictionCallback
from ..yolo_heads_predictions import YoloHeadsPredictions


@register_metric()
class RPYError(Metric):

    def __init__(
        self,
        post_prediction_callback: YoloHeadsPostPredictionCallback,
        min_iou: float = 0.5,
        threshold: float = 0.05,
        below: bool = True,
    ):
        super().__init__(
            compute_on_step=False,
            dist_sync_on_step=False,
        )
        self.post_prediction_callback = post_prediction_callback
        self.threshold = threshold
        self.below = below
        self.min_iou = min_iou
        self.add_state("roll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pitch", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("yaw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.component_names = ["RPY_roll", "RPY_pitch", "RPY_yaw", "RPY_mean"]
        self.greater_component_is_better = {"RPY_roll": False, "RPY_pitch": False, "RPY_yaw": False, "RPY_mean": False}

    def update(
        self,
        preds: Any,
        target: Any,
        gt_samples: List[MeshEstimationSample],
    ) -> None:
        """
        Update state with predictions and targets.

        Args:
            pred_keypoints: (Tensor[B, C, dim]): predicted keypoints tensor
            gts: Dict of tensors:
                {'keypoints' : Tensor[B, C, dim], 'bboxes': Tensor[B, 4]}.
                The key 'bboxes' expected for dim=2.

        dim reflects 2D-3D mode.
        """
        predictions: List[YoloHeadsPredictions] = self.post_prediction_callback(preds)
        assert len(predictions) == len(gt_samples)
        for image_index in range(len(gt_samples)):
            pred_bboxes_xyxy = predictions[image_index].bboxes_xyxy
            pred_mmparams = predictions[image_index].mm_params

            true_bboxes_xywh = torch.from_numpy(gt_samples[image_index].bboxes_xywh)
            true_rotation_matrix = torch.from_numpy(gt_samples[image_index].rotation_matrix)

            match_result = match_head_boxes(
                pred_boxes_xyxy=pred_bboxes_xyxy,
                true_boxes_xyxy=xywh_to_xyxy(true_bboxes_xywh, image_shape=None),
                min_iou=self.min_iou,
            )

            for pred_index, true_index in match_result.tp_matches:
                pred_mmparams_i = pred_mmparams[pred_index:pred_index+1]
                flame = FlameParams.from_3dmm(pred_mmparams_i, FLAME_CONSTS)
                pred_rpy = self.calculate_rpy_from_flame(flame)
                true_rpy = self.calculate_rpy_from_rotation_mat(true_rotation_matrix[true_index])

                self.roll += self.mae(pred_rpy.roll, true_rpy.roll)
                self.pitch += self.mae(pred_rpy.pitch, true_rpy.pitch)
                self.yaw += self.mae(pred_rpy.yaw, true_rpy.yaw)
                self.total_tp += 1.0

            total = len(match_result.fp_indexes) + len(match_result.fn_indexes) + len(match_result.tp_matches)
            self.total += float(total)

    def compute(self):
        total = int(self.total)
        total_tp = int(self.total_tp)
        if total_tp == 0:
            return {
                "RPY_roll": 100,
                "RPY_pitch": 100,
                "RPY_yaw": 100,
                "RPY_mean": 100,
            }
        else:
            acc = total_tp / total
            roll = (self.roll / total_tp) / acc
            pitch = (self.pitch / total_tp) / acc
            yaw = (self.yaw / total_tp) / acc

        return {
            "RPY_roll": float(roll),
            "RPY_pitch": float(pitch),
            "RPY_yaw": float(yaw),
            "RPY_mean": float(roll + pitch + yaw) / 3,
        }

    @staticmethod
    def calculate_rpy_from_flame(flame_params) -> RPY:
        rot_mat = rot_mat_from_6dof(flame_params.rotation).numpy()[0]
        return RPYError.calculate_rpy_from_rotation_mat(rot_mat)

    @staticmethod
    def calculate_rpy_from_rotation_mat(rot_mat):
        rot_mat_2 = np.transpose(rot_mat)
        angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
        roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
        return RPY(roll=roll, pitch=pitch, yaw=yaw)

    @staticmethod
    def mae(x, y):
        PI = 180.0
        return min(
            math.fabs(x - y),
            math.fabs(x - (y - 2 * PI)),
            math.fabs(x - (y + 2 * PI)),
        )


MAX_ROTATION = 99
MAE_THRESHOLD = 40


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
