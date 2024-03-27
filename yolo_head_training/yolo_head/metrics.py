import dataclasses
from typing import Callable, Any, Dict, Union, List, Tuple

import numpy as np
import torch
import torch.nn
from scipy.optimize import linear_sum_assignment
from super_gradients.common.registry import register_metric
from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback, PoseEstimationPredictions
from super_gradients.training.metrics.pose_estimation_utils import compute_oks
from super_gradients.training.samples import PoseEstimationSample
from torch import Tensor
from torchmetrics import Metric

__all__ = ["KeypointsFailureRate", "KeypointsNME"]


@dataclasses.dataclass
class PosesMatchingResult:
    tp_matches: List[Tuple[int, int]]
    fp_indexes: List[int]
    fn_indexes: List[int]


def match_poses(
        pred_poses: np.ndarray, true_poses: np.ndarray, true_bboxes_xywh: np.ndarray, min_oks: float,
        oks_sigmas: np.ndarray
) -> PosesMatchingResult:
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
    iou: np.ndarray = compute_oks(
        pred_joints=pred_poses,
        gt_joints=true_poses[..., :-1],
        gt_bboxes=true_bboxes_xywh,
        gt_keypoint_visibility=true_poses[..., -1],
        sigmas=oks_sigmas,
    ).cpu().numpy()

    # One gt pose can be matched to one pred pose and best match (in terms of IoU) is selected.
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)
    tp_matches = []
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= min_oks:
            tp_matches.append((r, c))
    fp_indexes = [i for i in range(pred_poses.shape[0]) if i not in col_ind]
    fn_indexes = [i for i in range(true_poses.shape[0]) if i not in row_ind]
    return PosesMatchingResult(tp_matches=tp_matches, fp_indexes=fp_indexes, fn_indexes=fn_indexes)


def metrics_w_bbox_wrapper(
        outputs: Tensor, gts: Union[Tensor, Dict[str, Tensor]], function: Callable, *args: Any, **kwargs: Any
) -> Tensor:
    gt_bboxes = gts["bboxes"] if "bboxes" in gts.keys() else None
    gt_keypoints = gts["keypoints"]
    if not torch.is_tensor(gt_bboxes):
        gt_bboxes = torch.from_numpy(gt_bboxes)
    if not torch.is_tensor(gt_keypoints):
        gt_keypoints = torch.from_numpy(gt_keypoints)
    return function(outputs, gt_keypoints, gt_bboxes, *args, **kwargs)


def keypoints_nme(
        output_kp: Tensor,
        target_kp: Tensor,
        bboxes_xywh: Tensor = None,
        reduce: str = "mean",
) -> Tensor:
    """
    https://arxiv.org/pdf/1708.07517v2.pdf
    """
    err = (output_kp - target_kp).norm(2, -1).mean(-1)
    # norm_distance = 2.0 for the 3D case, where the keypoints are in the normalized cube [-1; 1] ^ 3.
    norm_distance = torch.sqrt(bboxes_xywh[2] * bboxes_xywh[3]) if bboxes_xywh is not None else 2.0
    nme = torch.div(err, norm_distance)
    if reduce == "mean":
        nme = torch.mean(nme)
    return nme


def percentage_of_errors_below_IOD(
        output_kp: Tensor,
        target_kp: Tensor,
        bbox: Tensor = None,
        threshold: float = 0.05,
        below: bool = True,
) -> Tensor:
    """
    https://arxiv.org/pdf/1708.07517v2.pdf
    """
    err = (output_kp - target_kp).norm(2, -1).mean(-1)
    # norm_distance = 2.0 for the 3D case, where the keypoints are in the normalized cube [-1; 1] ^ 3.
    norm_distance = torch.sqrt(bbox[2] * bbox[3]) if bbox is not None else 2.0
    number_of_images = (err < threshold * norm_distance).sum() if below else (err > threshold * norm_distance).sum()
    return number_of_images  # percentage of such examples in a batch


@register_metric()
class KeypointsFailureRate(Metric):
    """Compute the Failure Rate metric for [2/3]D keypoints with averaging across individual examples."""

    def __init__(
            self,
            post_prediction_callback: AbstractPoseEstimationPostPredictionCallback,
            oks_sigmas: np.ndarray,
            oks_threshold: float = 0.05,
            threshold: float = 0.05,
            below: bool = True,
    ):
        super().__init__(
            dist_sync_on_step=False,
        )
        self.post_prediction_callback = post_prediction_callback
        self.threshold = threshold
        self.oks_threshold = float(oks_threshold)
        self.oks_sigmas = np.array(oks_sigmas)
        self.below = below
        self.add_state("failure_rate", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tp", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
            self,
            preds: Any,
            target: Any,
            gt_samples: List[PoseEstimationSample],
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
        predictions: List[PoseEstimationPredictions] = self.post_prediction_callback(preds)
        assert len(predictions) == len(gt_samples)
        for image_index in range(len(gt_samples)):
            pred_keypoints = predictions[image_index].poses.cpu()
            true_keypoints = gt_samples[image_index].joints  # All but last column
            true_bboxes_xywh = gt_samples[image_index].bboxes_xywh
            match_result = match_poses(
                pred_poses=pred_keypoints,
                true_poses=true_keypoints,
                true_bboxes_xywh=true_bboxes_xywh,
                min_oks=self.oks_threshold,
                oks_sigmas=self.oks_sigmas,
            )

            for pred_index, true_index in match_result.tp_matches:
                self.failure_rate += metrics_w_bbox_wrapper(
                    function=percentage_of_errors_below_IOD,
                    outputs=pred_keypoints[pred_index][..., 0:2],
                    gts={
                        "keypoints": true_keypoints[true_index][..., 0:2],
                        "bboxes": gt_samples[image_index].bboxes_xywh[true_index],
                    },
                    threshold=self.threshold,
                    below=self.below,
                )
                self.total_tp += 1.0

            total = len(match_result.fp_indexes) + len(match_result.fn_indexes) + len(match_result.tp_matches)
            self.total += float(total)

    def compute(self) -> torch.Tensor:
        acc = self.total_tp / self.total
        failure_rate = self.failure_rate / self.total_tp
        return (failure_rate / acc) if acc > 0 else 1.0


@register_metric()
class KeypointsNME(Metric):
    """Compute the NME Metric for [2/3]D keypoints with averaging across individual examples.
    https://arxiv.org/pdf/1708.07517v2.pdf
    """

    def __init__(
            self,
            post_prediction_callback: AbstractPoseEstimationPostPredictionCallback,
            oks_sigmas: np.ndarray,
            oks_threshold: float = 0.05,
            weight: int = 100,
    ):
        super().__init__(
            compute_on_step=False,
            dist_sync_on_step=False,
        )

        self.weight = weight
        self.oks_threshold = float(oks_threshold)
        self.oks_sigmas = np.array(oks_sigmas)
        self.post_prediction_callback = post_prediction_callback
        self.add_state("nme", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tp", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
            self,
            preds: Any,
            target: Any,
            gt_samples: List[PoseEstimationSample],
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
        predictions: List[PoseEstimationPredictions] = self.post_prediction_callback(preds)
        assert len(predictions) == len(gt_samples)

        for image_index in range(len(gt_samples)):
            pred_keypoints = predictions[image_index].poses.cpu()
            true_keypoints = gt_samples[image_index].joints  # All but last column
            match_result = match_poses(
                pred_poses=pred_keypoints,
                true_poses=true_keypoints,
                true_bboxes_xywh=gt_samples[image_index].bboxes_xywh,
                min_oks=self.oks_threshold,
                oks_sigmas=self.oks_sigmas,
            )

            for pred_index, true_index in match_result.tp_matches:
                self.nme += metrics_w_bbox_wrapper(
                    function=keypoints_nme,
                    outputs=pred_keypoints[pred_index][..., 0:2],
                    gts={
                        "keypoints": true_keypoints[true_index][..., 0:2],
                        "bboxes": gt_samples[image_index].bboxes_xywh[true_index],
                    },
                )
                self.total_tp += 1.0

            total = len(match_result.fp_indexes) + len(match_result.fn_indexes) + len(match_result.tp_matches)
            self.total += float(total)

    def compute(self) -> torch.Tensor:
        """
        Computes accuracy over state.
        """
        acc = self.total_tp / self.total
        nme = self.weight * (self.nme / self.total_tp)
        return (nme / acc) if acc > 0 else self.weight
