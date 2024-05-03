from typing import List, Tuple

import torch
import torchvision
from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback
from torch import Tensor

from .flame import FLAMELayer, FLAME_CONSTS, reproject_spatial_vertices, get_445_keypoints_indexes
from .yolo_head_ndfl_heads import YoloHeadsDecodedPredictions, YoloHeadsRawOutputs
from .yolo_heads_predictions import YoloHeadsPredictions


class YoloHeadsPostPredictionCallback(AbstractPoseEstimationPostPredictionCallback):
    """
    A post-prediction callback for YoloNASPose model.
    Performs confidence thresholding, Top-K and NMS steps.
    """

    def __init__(
        self,
        confidence_threshold: float,
        nms_iou_threshold: float,
        pre_nms_max_predictions: int,
        post_nms_max_predictions: int,
    ):
        """
        :param confidence_threshold: Pose detection confidence threshold
        :param nms_iou_threshold:         IoU threshold for NMS step.
        :param pre_nms_max_predictions:   Number of predictions participating in NMS step
        :param post_nms_max_predictions:  Maximum number of boxes to return after NMS step
        """
        if post_nms_max_predictions > pre_nms_max_predictions:
            raise ValueError("post_nms_max_predictions must be less than pre_nms_max_predictions")

        super().__init__()
        self.pose_confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.pre_nms_max_predictions = pre_nms_max_predictions
        self.post_nms_max_predictions = post_nms_max_predictions
        self.flame = FLAMELayer(FLAME_CONSTS)
        self.indexes_subset = get_445_keypoints_indexes()

    @torch.no_grad()
    def __call__(self, outputs: Tuple[YoloHeadsDecodedPredictions, YoloHeadsRawOutputs]) -> List[YoloHeadsPredictions]:
        """
        Take YoloNASPose's predictions and decode them into usable pose predictions.

        :param outputs: Output of the model's forward() method
        :return:        List of decoded predictions for each image in the batch.
        """
        # First is model predictions, second element of tuple is logits for loss computation
        predictions: YoloHeadsDecodedPredictions = outputs[0]

        decoded_predictions: List[YoloHeadsPredictions] = []

        for pred_bboxes_xyxy, pred_bboxes_conf, pred_flame_params in zip(
            predictions.boxes_xyxy.detach().cpu().float(),
            predictions.scores.detach().cpu().float(),
            predictions.flame_params.detach().cpu().float(),
        ):
            # pred_bboxes [Anchors, 4] in XYXY format
            # pred_scores [Anchors, 1] confidence scores [0..1]
            # pred_flame_params [Anchors, Flame Params]
            # pred_3d_vertices [Anchors, Vertices, 3]

            pred_bboxes_conf = pred_bboxes_conf.squeeze(-1)  # [Anchors]
            conf_mask = pred_bboxes_conf >= self.pose_confidence_threshold  # [Anchors]

            pred_bboxes_conf = pred_bboxes_conf[conf_mask]
            pred_bboxes_xyxy = pred_bboxes_xyxy[conf_mask]
            pred_flame_params = pred_flame_params[conf_mask]

            # Filter all predictions by self.nms_top_k
            if pred_bboxes_conf.size(0) > self.pre_nms_max_predictions:
                topk_candidates = torch.topk(pred_bboxes_conf, k=self.pre_nms_max_predictions, largest=True, sorted=True)
                pred_bboxes_conf = pred_bboxes_conf[topk_candidates.indices]
                pred_bboxes_xyxy = pred_bboxes_xyxy[topk_candidates.indices]
                pred_flame_params = pred_flame_params[topk_candidates.indices]

            # NMS
            idx_to_keep = torchvision.ops.boxes.nms(boxes=pred_bboxes_xyxy, scores=pred_bboxes_conf, iou_threshold=self.nms_iou_threshold)

            final_bboxes = pred_bboxes_xyxy[idx_to_keep][: self.post_nms_max_predictions]  # [Instances, 4]
            final_scores = pred_bboxes_conf[idx_to_keep][: self.post_nms_max_predictions]  # [Instances, 1]
            final_params = pred_flame_params[idx_to_keep][: self.post_nms_max_predictions]  # [Instances, Flame Params]

            final_3d_pts = reproject_spatial_vertices(self.flame, final_params, to_2d=True, subset_indexes=self.indexes_subset)
            final_2d_pts = final_3d_pts[..., :2]

            p = YoloHeadsPredictions(
                scores=final_scores,
                bboxes_xyxy=final_bboxes,
                mm_params=final_params,
                predicted_3d_vertices=final_3d_pts.clone(),
                predicted_2d_vertices=final_2d_pts.clone(),
            )

            decoded_predictions.append(p)

        return decoded_predictions
