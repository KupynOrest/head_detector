import math
from functools import partial
from typing import Tuple, Callable

import torch
from torch import nn, Tensor

from super_gradients.common.registry import register_detection_module
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules import ConvBNReLU, QARepVGGBlock
from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.modules.utils import width_multiplier

from yolo_head.flame import FLAME_CONSTS


@register_detection_module()
class YoloHeadsDFLHead(BaseDetectionModule):
    """
    YoloNASPoseDFLHead is the head used in YoloNASPose model.
    This class implements single-class object detection and keypoints regression on a single scale feature map
    """

    def __init__(
        self,
        in_channels: int,
        bbox_inter_channels: int,
        flame_inter_channels: int,
        flame_shape_inter_channels: int,
        flame_expression_inter_channels: int,
        flame_transformation_inter_channels: int,
        flame_regression_blocks: int,
        shared_stem: bool,
        width_mult: float,
        first_conv_group_size: int,
        # num_classes: int,
        stride: int,
        reg_max: int,
        cls_dropout_rate: float = 0.0,
        reg_dropout_rate: float = 0.0,
    ):
        """
        Initialize the YoloNASDFLHead
        :param in_channels: Input channels
        :param bbox_inter_channels: Intermediate number of channels for box detection & regression
        :param flame_inter_channels: Intermediate number of channels for pose regression
        :param shared_stem: Whether to share the stem between the pose and bbox heads
        :param pose_conf_in_class_head: Whether to include the pose confidence in the classification head
        :param width_mult: Width multiplier
        :param first_conv_group_size: Group size
        :param num_classes: Number of keypoints classes for pose regression. Number of detection classes is always 1.
        :param stride: Output stride for this head
        :param reg_max: Number of bins in the regression head
        :param cls_dropout_rate: Dropout rate for the classification head
        :param reg_dropout_rate: Dropout rate for the regression head
        """
        super().__init__(in_channels)

        bbox_inter_channels = width_multiplier(bbox_inter_channels, width_mult, 8)
        flame_inter_channels = width_multiplier(flame_inter_channels, width_mult, 8)

        if first_conv_group_size == 0:
            groups = 0
        elif first_conv_group_size == -1:
            groups = 1
        else:
            groups = bbox_inter_channels // first_conv_group_size

        #self.num_classes = num_classes
        self.shared_stem = shared_stem

        if self.shared_stem:
            max_input = max(bbox_inter_channels, flame_inter_channels)
            self.stem = ConvBNReLU(in_channels, max_input, kernel_size=1, stride=1, padding=0, bias=False)

            if max_input != flame_inter_channels:
                self.pose_stem = nn.Conv2d(max_input, flame_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.pose_stem = nn.Identity()

            if max_input != bbox_inter_channels:
                self.bbox_stem = nn.Conv2d(max_input, bbox_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.bbox_stem = nn.Identity()

        else:
            self.stem = nn.Identity()
            self.pose_stem = ConvBNReLU(in_channels, flame_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bbox_stem = ConvBNReLU(in_channels, bbox_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        first_cls_conv = [ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.cls_convs = nn.Sequential(*first_cls_conv, ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_reg_conv = [ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.reg_convs = nn.Sequential(*first_reg_conv, ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.reg_pred = nn.Conv2d(bbox_inter_channels, 4 * (reg_max + 1), 1, 1, 0)

        self.cls_pred = nn.Conv2d(bbox_inter_channels, 1, 1, 1, 0)

        self.flame_shape_pred = self.build_flame_regression_layers(
            flame_inter_channels, flame_shape_inter_channels, flame_regression_blocks, FLAME_CONSTS["shape"]
        )
        self.flame_expression_pred = self.build_flame_regression_layers(
            flame_inter_channels, flame_expression_inter_channels, flame_regression_blocks, FLAME_CONSTS["expression"]
        )
        self.flame_rotation_pred = self.build_flame_regression_layers(
            flame_inter_channels, flame_transformation_inter_channels, flame_regression_blocks, FLAME_CONSTS["rotation"]
        )
        self.flame_jaw_pred = self.build_flame_regression_layers(
            flame_inter_channels, flame_transformation_inter_channels, flame_regression_blocks, FLAME_CONSTS["jaw"]
        )
        self.flame_scale_pred = self.build_flame_regression_layers(
            flame_inter_channels, flame_transformation_inter_channels, flame_regression_blocks, FLAME_CONSTS["scale"]
        )
        self.flame_translation_pred = self.build_flame_regression_layers(
            flame_inter_channels, flame_transformation_inter_channels, flame_regression_blocks, FLAME_CONSTS["translation"]
        )

        self.cls_dropout_rate = nn.Dropout2d(cls_dropout_rate) if cls_dropout_rate > 0 else nn.Identity()
        self.reg_dropout_rate = nn.Dropout2d(reg_dropout_rate) if reg_dropout_rate > 0 else nn.Identity()

        self.stride = stride

        self.prior_prob = 1e-2
        self._initialize_biases()

    def build_flame_regression_layers(self, in_channels, inter_channels, num_blocks, out_channels):
        pose_block = partial(QARepVGGBlock, use_residual_connection=False, use_alpha=True)
        layers = []
        for _ in range(num_blocks):
            layers.append(pose_block(in_channels, inter_channels))
            in_channels = inter_channels
        layers.append(nn.Conv2d(inter_channels, out_channels, 1, 1, 0))
        return nn.Sequential(*layers)

    @property
    def out_channels(self):
        return None

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor]:
        """

        :param x: Input feature map of shape [B, Cin, H, W]
        :return: Tuple of [reg_output, cls_output, pose_regression, pose_logits]
            - reg_output:      Tensor of [B, 4 * (reg_max + 1), H, W]
            - cls_output:      Tensor of [B, 1, H, W]
            - flame_output: Tensor of [B, Num Params, H, W]
        """
        x = self.stem(x)
        pose_features = self.pose_stem(x)
        bbox_features = self.bbox_stem(x)

        cls_feat = self.cls_convs(bbox_features)
        cls_feat = self.cls_dropout_rate(cls_feat)
        cls_output = self.cls_pred(cls_feat)

        reg_feat = self.reg_convs(bbox_features)
        reg_feat = self.reg_dropout_rate(reg_feat)
        reg_output = self.reg_pred(reg_feat)

        flame_shape = self.flame_shape_pred(pose_features).tanh() * 3
        flame_expression = self.flame_expression_pred(pose_features).tanh() * 3
        flame_rotation = self.flame_rotation_pred(pose_features)
        flame_jaw = self.flame_jaw_pred(pose_features)
        flame_translation = self.flame_translation_pred(pose_features)
        flame_scale = self.flame_scale_pred(pose_features).exp() * self.stride

        flame_output = torch.cat([flame_shape, flame_expression, flame_rotation, flame_jaw, flame_translation, flame_scale], dim=1)

        return reg_output, cls_output, flame_output

    def _initialize_biases(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_pred.bias, prior_bias)
