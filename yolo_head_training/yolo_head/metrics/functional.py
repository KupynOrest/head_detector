import dataclasses
from typing import Callable, Any, Dict, Union, List, Tuple

import numpy as np
import torch
import torch.nn
import torchvision
from scipy.optimize import linear_sum_assignment
from super_gradients.common.registry import register_metric
from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback, PoseEstimationPredictions
from super_gradients.training.metrics.pose_estimation_utils import compute_oks
from super_gradients.training.samples import PoseEstimationSample
from torch import Tensor
from torchmetrics import Metric


from yolo_head.flame import FLAMELayer, FLAME_CONSTS, FlameParams



@dataclasses.dataclass
class HeadsMatchingResult:
    tp_matches: List[Tuple[int, int]]
    fp_indexes: List[int]
    fn_indexes: List[int]


def match_head_boxes(pred_boxes_xyxy: Tensor, true_boxes_xyxy: Tensor, min_iou: float) -> HeadsMatchingResult:
    """
    Match two sets of head boxes based on the IoU metric.
    A TP match is a pair of indexes of a predicted box and a ground truth box, which are considered to be matched.
    A TP is a pair of boxes with IoU >= min_iou.

    Args:
        pred_boxes_xyxy: [N, 4] tensor of predicted boxes in XYXY format.
        true_boxes_xyxy: [M, 4] tensor of ground truth boxes in XYXY format.
        min_iou: minimum IoU value for a match.

    Returns:
        PosesMatchingResult object.
    """
    iou: Tensor = torchvision.ops.box_iou(pred_boxes_xyxy, true_boxes_xyxy).cpu().numpy()

    # One gt box can be matched to one pred box and best match (in terms of IoU) is selected.
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)
    tp_matches = []
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= min_iou:
            tp_matches.append((r, c))
    fp_indexes = [i for i in range(pred_boxes_xyxy.shape[0]) if i not in col_ind]
    fn_indexes = [i for i in range(true_boxes_xyxy.shape[0]) if i not in row_ind]
    return HeadsMatchingResult(tp_matches=tp_matches, fp_indexes=fp_indexes, fn_indexes=fn_indexes)

def match_poses(pred_poses: np.ndarray, true_poses: np.ndarray, true_bboxes_xywh: np.ndarray, min_oks: float, oks_sigmas: np.ndarray) -> HeadsMatchingResult:
    """
    Match two sets of poses based on the OKS metric.
    A tp match is a pair of indexes of a predicted pose and a ground truth pose, which are considered to be matched.
    A TP is a pair of poses with OKS >= min_oks.

    Args:
        pred_poses: [N, NumKeypoints, 2/3] array of predicted poses.
        true_poses: [M, NumKeypoints, 3/4] array of ground truth poses.
        min_oks: minimum OKS value for a match.
        oks_sigmas: [NumKeypoints] array of sigmas for OKS calculation.

    Returns:

    """
    # Compute OKS matrix
    if not torch.is_tensor(pred_poses):
        pred_poses = torch.from_numpy(pred_poses)
    if not torch.is_tensor(true_poses):
        true_poses = torch.from_numpy(true_poses)
    if not torch.is_tensor(oks_sigmas):
        oks_sigmas = torch.from_numpy(oks_sigmas)
    if not torch.is_tensor(true_bboxes_xywh):
        true_bboxes_xywh = torch.from_numpy(true_bboxes_xywh)
    iou: np.ndarray = (
        compute_oks(
            pred_joints=pred_poses,
            gt_joints=true_poses[..., :-1],
            gt_bboxes=true_bboxes_xywh,
            gt_keypoint_visibility=true_poses[..., -1],
            sigmas=oks_sigmas,
        )
        .cpu()
        .numpy()
    )

    # One gt pose can be matched to one pred pose and best match (in terms of IoU) is selected.
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)
    tp_matches = []
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= min_oks:
            tp_matches.append((r, c))
    fp_indexes = [i for i in range(pred_poses.shape[0]) if i not in col_ind]
    fn_indexes = [i for i in range(true_poses.shape[0]) if i not in row_ind]
    return HeadsMatchingResult(tp_matches=tp_matches, fp_indexes=fp_indexes, fn_indexes=fn_indexes)


def metrics_w_bbox_wrapper(outputs: Tensor, gts: Union[Tensor, Dict[str, Tensor]], function: Callable, *args: Any, **kwargs: Any) -> Tensor:
    gt_bboxes = gts["bboxes"] if "bboxes" in gts.keys() else None
    gt_keypoints = gts["keypoints"]
    if not torch.is_tensor(gt_bboxes):
        gt_bboxes = torch.from_numpy(gt_bboxes)
    if not torch.is_tensor(gt_keypoints):
        gt_keypoints = torch.from_numpy(gt_keypoints)
    return function(outputs, gt_keypoints, gt_bboxes, *args, **kwargs)
