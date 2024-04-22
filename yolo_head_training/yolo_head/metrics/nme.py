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


from yolo_head.flame import FLAMELayer, FLAME_CONSTS, FlameParams

from .functional import match_poses, metrics_w_bbox_wrapper

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

@register_metric()
class KeypointsNME(Metric):
    """Compute the NME Metric for [2/3]D keypoints with averaging across individual examples.
    https://arxiv.org/pdf/1708.07517v2.pdf
    """

    def __init__(
        self,
        post_prediction_callback: AbstractPoseEstimationPostPredictionCallback,
        weight: int = 100,
    ):
        super().__init__(
            compute_on_step=False,
            dist_sync_on_step=False,
        )

        self.weight = weight
        self.post_prediction_callback = post_prediction_callback
        self.flame = FLAMELayer(FLAME_CONSTS)
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
            mm_paramers = predictions[image_index].poses.cpu() # [N, NumParams]
            flame_params = FlameParams.from_3dmm(mm_paramers, FLAME_CONSTS)
            vertices3d = self.flame(flame_params)


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
        return (nme / acc) if acc > 0 else torch.tensor(self.weight, dtype=torch.float32, device=self.device)
