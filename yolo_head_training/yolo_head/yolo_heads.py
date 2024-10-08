from typing import Union, Optional, Tuple, Any

import torch
from torch import Tensor
from omegaconf import DictConfig
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.common.registry import register_model
from super_gradients.module_interfaces import SupportsInputShapeCheck
from super_gradients.training.models import CustomizableDetector
from super_gradients.training.processing.processing import Processing
from super_gradients.training.utils import HpmStruct

from yolo_head.yolo_heads_post_prediction_callback import YoloHeadsPostPredictionCallback
from yolo_head.exportable_mesh_model import ExportableMeshEstimationModel, AbstractMeshDecodingModule


class VGGHeadDecodingModule(AbstractMeshDecodingModule):
    __constants__ = ["num_pre_nms_predictions"]

    def __init__(
        self,
        num_pre_nms_predictions: int = 1000,
    ):
        super().__init__()
        self.num_pre_nms_predictions = num_pre_nms_predictions
        self.register_buffer('max_batch', torch.arange(8192))

    @torch.jit.ignore
    def infer_total_number_of_predictions(self, inputs: Any) -> int:
        """

        :param inputs: YoloNASPose model outputs
        :return:
        """
        predictions = inputs[0]
        pred_bboxes_xyxy = predictions.boxes_xyxy

        return pred_bboxes_xyxy.size(1)

    def get_num_pre_nms_predictions(self) -> int:
        return self.num_pre_nms_predictions

    def forward(self, inputs: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, ...]]):
        """
        Decode YoloNASPose model outputs into bounding boxes, confidence scores and pose coordinates and scores

        :param inputs: YoloNASPose model outputs
        :return: Tuple of (pred_bboxes, pred_scores, pred_joints)
        - pred_bboxes: [Batch, num_pre_nms_predictions, 4] Bounding of associated with pose in XYXY format
        - pred_scores: [Batch, num_pre_nms_predictions, 1] Confidence scores [0..1] for entire pose
        - pred_joints: [Batch, num_pre_nms_predictions, Num Joints, 3] Joints in (x,y,confidence) format
        """
        if torch.jit.is_tracing():
            pred_bboxes_xyxy, pred_bboxes_conf, flame_params = inputs
        else:
            predictions = inputs[0]

            pred_bboxes_xyxy = predictions.boxes_xyxy
            pred_bboxes_conf = predictions.scores
            flame_params = predictions.flame_params

        nms_top_k = self.num_pre_nms_predictions
        batch_size, num_anchors, _ = pred_bboxes_conf.size()

        topk_candidates = torch.topk(pred_bboxes_conf, dim=1, k=nms_top_k, largest=True, sorted=True)

        if torch.jit.is_tracing():
            batch_size = pred_bboxes_conf.shape[0]
            offsets = num_anchors * self.max_batch[:batch_size]
        else:
            offsets = num_anchors * torch.arange(batch_size, device=pred_bboxes_conf.device)
        indices_with_offset = topk_candidates.indices + offsets.reshape(batch_size, 1, 1)
        flat_indices = torch.flatten(indices_with_offset)

        output_pred_bboxes = pred_bboxes_xyxy.reshape(-1, pred_bboxes_xyxy.size(2))[flat_indices, :].reshape(
            pred_bboxes_xyxy.size(0), nms_top_k, pred_bboxes_xyxy.size(2)
        )
        output_pred_scores = pred_bboxes_conf.reshape(-1, pred_bboxes_conf.size(2))[flat_indices, :].reshape(
            pred_bboxes_conf.size(0), nms_top_k, pred_bboxes_conf.size(2)
        )
        output_pred_flame = flame_params.reshape(-1, flame_params.size(2))[flat_indices, :].reshape(
            flame_params.size(0), nms_top_k, flame_params.size(2)
        )

        return output_pred_bboxes, output_pred_scores, output_pred_flame


@register_model()
class YoloHeads(CustomizableDetector, ExportableMeshEstimationModel, SupportsInputShapeCheck):

    def __init__(
        self,
        backbone: Union[str, dict, HpmStruct, DictConfig],
        heads: Union[str, dict, HpmStruct, DictConfig],
        neck: Optional[Union[str, dict, HpmStruct, DictConfig]] = None,
        num_classes: int = None,
        bn_eps: Optional[float] = None,
        bn_momentum: Optional[float] = None,
        inplace_act: Optional[bool] = True,
        in_channels: int = 3,
    ):
        super().__init__(
            backbone=backbone,
            heads=heads,
            neck=neck,
            num_classes=num_classes,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            inplace_act=inplace_act,
            in_channels=in_channels,
        )
        self._edge_links = None
        self._edge_colors = None
        self._keypoint_colors = None
        self._image_processor = None
        self._default_nms_conf = None
        self._default_nms_iou = None
        self._default_pre_nms_max_predictions = None
        self._default_post_nms_max_predictions = None

    def get_decoding_module(self, num_pre_nms_predictions: int, **kwargs) -> AbstractMeshDecodingModule:
        return VGGHeadDecodingModule(num_pre_nms_predictions)

    @classmethod
    def get_post_prediction_callback(
        cls, conf: float, iou: float, pre_nms_max_predictions=1000, post_nms_max_predictions=300
    ) -> YoloHeadsPostPredictionCallback:
        return YoloHeadsPostPredictionCallback(
            confidence_threshold=conf,
            nms_iou_threshold=iou,
            pre_nms_max_predictions=pre_nms_max_predictions,
            post_nms_max_predictions=post_nms_max_predictions,
        )

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare the model to be converted to ONNX or other frameworks.
        Typically, this function will freeze the size of layers which is otherwise flexible, replace some modules
        with convertible substitutes and remove all auxiliary or training related parts.
        :param input_size: [H,W]
        """
        h, w = input_size[-2:]
        self.heads.cache_anchors((h, w))

    def get_preprocessing_callback(self, **kwargs):
        processing = self.get_processing_params()
        preprocessing_module = processing.get_equivalent_photometric_module()
        return preprocessing_module

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(
        self,
        image_processor: Optional[Processing] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = 0.7,
        pre_nms_max_predictions=300,
        post_nms_max_predictions=100,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        :param conf:            (Optional) Below the confidence threshold, prediction are discarded
        """
        self._image_processor = image_processor or self._image_processor
        self._default_nms_conf = conf or self._default_nms_conf
        self._default_nms_iou = iou or self._default_nms_iou
        self._default_pre_nms_max_predictions = pre_nms_max_predictions or self._default_pre_nms_max_predictions
        self._default_post_nms_max_predictions = post_nms_max_predictions or self._default_post_nms_max_predictions

    def get_input_shape_steps(self) -> Tuple[int, int]:
        """
        Returns the minimum input shape size that the model can accept.
        For segmentation models the default is 32x32, which corresponds to the largest stride in the encoder part of the model
        """
        return 32, 32

    def get_minimum_input_shape_size(self) -> Tuple[int, int]:
        """
        Returns the minimum input shape size that the model can accept.
        For segmentation models the default is 32x32, which corresponds to the largest stride in the encoder part of the model
        """
        return 32, 32
