from typing import Any, List

import torch
import torch.nn
from super_gradients.common.registry import register_metric
from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy
from super_gradients.training.samples import PoseEstimationSample
from torch import Tensor
from torchmetrics import Metric

from yolo_head.flame import FLAMELayer, FLAME_CONSTS
from .functional import metrics_w_bbox_wrapper, match_head_boxes
from .. import YoloHeadsPostPredictionCallback
from ..yolo_heads_predictions import YoloHeadsPredictions


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
        post_prediction_callback: YoloHeadsPostPredictionCallback,
        min_iou: float = 0.5,
        weight: int = 100,
    ):
        super().__init__(
            compute_on_step=False,
            dist_sync_on_step=False,
        )

        self.weight = weight
        self.post_prediction_callback = post_prediction_callback
        self.min_iou = min_iou
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
        predictions: List[YoloHeadsPredictions] = self.post_prediction_callback(preds)
        assert len(predictions) == len(gt_samples)
        for image_index in range(len(gt_samples)):
            # pred_mm_params = predictions[image_index].mm_params.cpu()
            pred_bboxes_xyxy = predictions[image_index].bboxes_xyxy
            # pred_vertices_3d = predictions[image_index].predicted_3d_vertices.cpu()
            pred_vertices_2d = predictions[image_index].predicted_2d_vertices.cpu()

            true_bboxes_xywh = torch.from_numpy(gt_samples[image_index].bboxes_xywh)
            true_keypoints = torch.from_numpy(gt_samples[image_index].joints)

            match_result = match_head_boxes(
                pred_boxes_xyxy=pred_bboxes_xyxy,
                true_boxes_xyxy=xywh_to_xyxy(true_bboxes_xywh, image_shape=None),
                min_iou=self.min_iou,
            )

            for pred_index, true_index in match_result.tp_matches:
                self.nme += metrics_w_bbox_wrapper(
                    function=keypoints_nme,
                    outputs=pred_vertices_2d[pred_index][..., 0:2],
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
