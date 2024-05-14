from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from super_gradients.common.registry import register_callback
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy
from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.utils.callbacks.callbacks import ExtremeBatchCaseVisualizationCallback
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
from torch import Tensor
from torchmetrics import Metric

from .yolo_heads_post_prediction_callback import YoloHeadsPostPredictionCallback
from .flame import get_indices
from .yolo_heads_predictions import YoloHeadsPredictions


@register_callback()
class ExtremeBatchYoloHeadsVisualizationCallback(ExtremeBatchCaseVisualizationCallback):

    def __init__(
        self,
        indexes_subset: Union[None, float, str],
        post_prediction_callback: YoloHeadsPostPredictionCallback,
        metric: Optional[Metric] = None,
        metric_component_name: Optional[str] = None,
        loss_to_monitor: Optional[str] = None,
        max: bool = False,
        freq: int = 1,
        max_images: Optional[int] = None,
        enable_on_train_loader: bool = False,
        enable_on_valid_loader: bool = True,
        keypoints_color: Tuple[int, int, int] = (0, 255, 255),
    ):
        super().__init__(
            metric=metric,
            metric_component_name=metric_component_name,
            loss_to_monitor=loss_to_monitor,
            max=max,
            freq=freq,
            enable_on_train_loader=enable_on_train_loader,
            enable_on_valid_loader=enable_on_valid_loader,
        )
        self.post_prediction_callback = post_prediction_callback
        self.keypoints_color = tuple(keypoints_color)
        self.max_images = max_images
        self.indexes_subset = torch.tensor(get_indices()[indexes_subset]).long() if indexes_subset is not None else None

    @classmethod
    def universal_undo_preprocessing_fn(cls, inputs: torch.Tensor) -> np.ndarray:
        """
        A universal reversing of preprocessing to be passed to DetectionVisualization.visualize_batch's undo_preprocessing_func kwarg.
        :param inputs:
        :return:
        """
        inputs = inputs - inputs.min()
        inputs /= inputs.max()
        inputs *= 255
        inputs = inputs.to(torch.uint8)
        inputs = inputs.cpu().numpy()
        inputs = inputs[:, ::-1, :, :].transpose(0, 2, 3, 1)
        inputs = np.ascontiguousarray(inputs, dtype=np.uint8)
        return inputs

    def _visualize_batch(
        self,
        image_tensor: np.ndarray,
        keypoints: List[Union[np.ndarray, Tensor]],
        bboxes: List[Union[None, np.ndarray, Tensor]],
        scores: Optional[List[Union[None, np.ndarray, Tensor]]],
        is_crowd: Optional[List[Union[None, np.ndarray, Tensor]]],
        keypoints_color: Tuple[int, int, int],
    ) -> List[np.ndarray]:
        """
        Generate list of samples visualization of a batch of images with keypoints and bounding boxes.

        :param image_tensor:             Images batch of [Batch Size, 3, H, W] shape with values in [0, 255] range.
                                         The images should be scaled to [0, 255] range and converted to uint8 type beforehead.
        :param keypoints:                Keypoints in XY format. Shape [Num Instances, Num Joints, 2]. Can be None.
        :param bboxes:                   Bounding boxes in XYXY format. Shape [Num Instances, 4]. Can be None.
        :param scores:                   Keypoint scores. Shape [Num Instances, Num Joints]. Can be None.
        :param is_crowd:                 Whether each sample is crowd or not. Shape [Num Instances]. Can be None.
        :param keypoints_color:          Keypoint colors. Shape [3]
        :return:                         List of visualization images.
        """

        out_images = []
        for i in range(image_tensor.shape[0]):
            keypoints_i = keypoints[i]
            bboxes_i = bboxes[i]
            scores_i = scores[i] if scores is not None else None
            is_crowd_i = is_crowd[i] if is_crowd is not None else None

            if self.indexes_subset is not None:
                keypoints_i = keypoints_i[..., self.indexes_subset, :]

            if torch.is_tensor(keypoints_i):
                keypoints_i = keypoints_i.detach().cpu().numpy()
            if torch.is_tensor(bboxes_i):
                bboxes_i = bboxes_i.detach().cpu().numpy()
            if torch.is_tensor(scores_i):
                scores_i = scores_i.detach().cpu().numpy()
            if torch.is_tensor(is_crowd_i):
                is_crowd_i = is_crowd_i.detach().cpu().numpy()

            num_joints = keypoints_i.shape[1]
            res_image = image_tensor[i]
            res_image = PoseVisualization.draw_poses(
                image=res_image,
                poses=keypoints_i,
                boxes=bboxes_i,
                scores=scores_i,
                is_crowd=is_crowd_i,
                show_keypoint_confidence=False,
                edge_links=[],
                edge_colors=[],
                keypoint_colors=[keypoints_color] * num_joints,
                keypoint_confidence_threshold=0.01,
            )

            out_images.append(res_image)

        return out_images

    @torch.no_grad()
    def process_extreme_batch(self) -> np.ndarray:
        """
        Processes the extreme batch, and returns batche of images for visualization - predictions and GT poses stacked horizontally.

        :return: np.ndarray - the visualization of predictions and GT
        """
        if "gt_samples" not in self.extreme_additional_batch_items:
            raise RuntimeError(
                "ExtremeBatchPoseEstimationVisualizationCallback requires 'gt_samples' to be present in additional_batch_items."
                "Currently only YoloNASPose model is supported. Old DEKR recipe is not supported at the moment."
            )

        inputs = self.universal_undo_preprocessing_fn(self.extreme_batch)
        gt_samples: List[PoseEstimationSample] = self.extreme_additional_batch_items["gt_samples"]
        predictions: List[YoloHeadsPredictions] = self.post_prediction_callback(self.extreme_preds)

        images_to_save_preds = self._visualize_batch(
            image_tensor=inputs,
            keypoints=[p.predicted_2d_vertices for p in predictions],
            bboxes=[p.bboxes_xyxy for p in predictions],
            scores=[p.scores for p in predictions],
            is_crowd=None,
            keypoints_color=self.keypoints_color,
        )
        images_to_save_preds = np.stack(images_to_save_preds)

        images_to_save_gt = self._visualize_batch(
            image_tensor=inputs,
            keypoints=[gt.joints for gt in gt_samples],
            bboxes=[xywh_to_xyxy(gt.bboxes_xywh, image_shape=None) for gt in gt_samples],
            scores=None,
            is_crowd=[gt.is_crowd for gt in gt_samples],
            keypoints_color=self.keypoints_color,
        )
        images_to_save_gt = np.stack(images_to_save_gt)

        # Stack the predictions and GT images together
        return np.concatenate([images_to_save_gt, images_to_save_preds], axis=2)
