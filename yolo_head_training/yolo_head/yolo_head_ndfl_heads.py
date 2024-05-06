import dataclasses
from typing import Tuple, Union, List, Callable

import einops
import super_gradients.common.factories.detection_modules_factory as det_factory
import torch
from omegaconf import DictConfig
from super_gradients.common.registry import register_detection_module
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import generate_anchors_for_grid_cell
from super_gradients.training.utils import HpmStruct, torch_version_is_greater_or_equal
from super_gradients.training.utils.bbox_utils import batch_distance2bbox
from torch import nn, Tensor

from yolo_head.flame import FLAME_CONSTS, FlameParams


@dataclasses.dataclass
class YoloHeadsDecodedPredictions:
    """
    A data class describing the decoded predictions from the YoloHeads module.
    :param boxes_xyxy: [B, Anchors, 4] Predicted boxes in XYXY format
    :param scores: [B, Anchors, 4] Predicted scores for each box
    :param flame_params: [B, Anchors, Flame Params] Predicted flame parameters
    """

    boxes_xyxy: Tensor
    scores: Tensor
    flame_params: Tensor


@dataclasses.dataclass
class YoloHeadsRawOutputs:
    """
    :params flame_params: [B, Anchors, Num Flame Params]
    :param flame_params: [B, Anchors, Flame Params] Predicted flame parameters
    """

    cls_score_list: Tensor
    reg_distri_list: Tensor
    flame_params: Tensor
    anchors: Tensor
    anchor_points: Tensor
    num_anchors_list: List[int]
    stride_tensor: Tensor


@register_detection_module()
class YoloHeadsNDFLHeads(BaseDetectionModule, SupportsReplaceNumClasses):
    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        heads_list: List[Union[HpmStruct, DictConfig]],
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
        reg_max: int = 16,
        width_mult: float = 1.0,
    ):
        """
        Initializes the NDFLHeads module.

        :param in_channels: Number of channels for each feature map (See width_mult)
        :param grid_cell_scale: A scaling factor applied to the grid cell coordinates.
               This scaling factor is used to define anchor boxes (see generate_anchors_for_grid_cell).
        :param grid_cell_offset: A fixed offset that is added to the grid cell coordinates.
               This offset represents a 'center' of the cell and is 0.5 by default.
        :param reg_max: Number of bins in the regression head
        :param width_mult: A scaling factor applied to in_channels.


        """
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        super().__init__(in_channels)

        self.in_channels = tuple(in_channels)
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max

        # Do not apply quantization to this tensor
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj, persistent=False)

        factory = det_factory.DetectionModulesFactory()
        # heads_list = self._insert_heads_list_params(heads_list, factory, num_classes, reg_max)

        self.num_heads = len(heads_list)
        fpn_strides: List[int] = []
        for i in range(self.num_heads):
            new_head = factory.get(factory.insert_module_param(heads_list[i], "in_channels", in_channels[i]))
            fpn_strides.append(new_head.stride)
            setattr(self, f"head{i + 1}", new_head)

        self.fpn_strides = tuple(fpn_strides)

    @staticmethod
    def _insert_heads_list_params(
        heads_list: List[Union[HpmStruct, DictConfig]], factory: det_factory.DetectionModulesFactory, reg_max: int
    ) -> List[Union[HpmStruct, DictConfig]]:
        """
        Injects num_classes and reg_max parameters into the heads_list.

        :param heads_list:  Input heads list
        :param factory:     DetectionModulesFactory
        :param num_classes: Number of classes
        :param reg_max:     Number of bins in the regression head
        :return:            Heads list with injected parameters
        """
        for i in range(len(heads_list)):
            # heads_list[i] = factory.insert_module_param(heads_list[i], "num_classes", num_classes)
            heads_list[i] = factory.insert_module_param(heads_list[i], "reg_max", reg_max)
        return heads_list

    def forward(self, feats: Tuple[Tensor, ...]) -> Union[YoloHeadsDecodedPredictions, Tuple[YoloHeadsDecodedPredictions, YoloHeadsRawOutputs]]:
        """
        Runs the forward for all the underlying heads and concatenate the predictions to a single result.
        :param feats: List of feature maps from the neck of different strides
        :return: Return value depends on the mode:
        If tracing, a tuple of 4 tensors (decoded predictions) is returned:
        - pred_bboxes [B, Num Anchors, 4] - Predicted boxes in XYXY format
        - pred_scores [B, Num Anchors, 1] - Predicted scores for each box
        - pred_pose_coords [B, Num Anchors, Num Keypoints, 2] - Predicted poses in XY format
        - pred_pose_scores [B, Num Anchors, Num Keypoints] - Predicted scores for each keypoint

        In training/eval mode, a tuple of 2 tensors returned:
        - decoded predictions - they are the same as in tracing mode
        - raw outputs - a tuple of 8 elements in total, this is needed for training the model.
        """

        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []
        flame_params_list = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            stride = self.fpn_strides[i]

            height_mul_width = h * w
            reg_distri, cls_logit, flame_outputs = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = torch.nn.functional.softmax(reg_dist_reduced, dim=1).mul(self.proj_conv).sum(1)

            # cls and reg
            cls_score_list.append(cls_logit.reshape([b, -1, height_mul_width]))
            reg_dist_reduced_list.append(reg_dist_reduced)

            flame_params_list.append(flame_outputs.flatten(2))  # [B, Num Flame Params, H, W] -> [B, Num Flame Params, H * W]

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [B, C, Anchors]
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])  # # [B, Anchors, C]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 4 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)  # [B, Anchors, 4]

        anchor_points_inference, stride_tensor = self._generate_anchors(feats)
        centers = anchor_points_inference * stride_tensor

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor  # [B, Anchors, 4]

        flame_params_list = torch.cat(flame_params_list, dim=-1)  # [B, Num Flame Params, Anchors]
        flame_params = FlameParams.from_3dmm(flame_params_list, FLAME_CONSTS)
        flame_params.translation[:, 0:2] += einops.rearrange(centers, "A N -> 1 N A")
        flame_params = flame_params.to_3dmm_tensor()  # [B, Num Flame Params, Anchors]
        flame_params = einops.rearrange(flame_params, "B F A -> B A F")  # Rearrange to common format where anchors comes first

        decoded_predictions = YoloHeadsDecodedPredictions(boxes_xyxy=pred_bboxes, scores=pred_scores, flame_params=flame_params)

        anchors, anchor_points, num_anchors_list, _ = generate_anchors_for_grid_cell(feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset)

        raw_predictions = YoloHeadsRawOutputs(
            cls_score_list=cls_score_list,
            reg_distri_list=reg_distri_list,
            flame_params=flame_params,
            anchors=anchors,
            anchor_points=anchor_points,
            num_anchors_list=num_anchors_list,
            stride_tensor=stride_tensor,
        )
        return decoded_predictions, raw_predictions

    @property
    def out_channels(self):
        return None

    def _generate_anchors(self, feats=None, dtype=None, device=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []

        dtype = dtype or feats[0].dtype
        device = device or feats[0].device

        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            if torch_version_is_greater_or_equal(1, 10):
                shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            else:
                shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=dtype))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)

        if device is not None:
            anchor_points = anchor_points.to(device)
            stride_tensor = stride_tensor.to(device)
        return anchor_points, stride_tensor
