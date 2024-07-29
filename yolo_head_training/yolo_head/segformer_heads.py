import copy
from pathlib import Path
from typing import Union

from omegaconf import DictConfig
from super_gradients.common.factories.detection_modules_factory import DetectionModulesFactory
from super_gradients.common.registry import register_model
from super_gradients.training.models.segmentation_models.segformer import MiTBackBone
from torch import nn
from super_gradients.common.registry import register_model
from super_gradients.training.models import get_arch_params
from super_gradients.training.utils import HpmStruct, get_param
from super_gradients.training.models import SgModule


@register_model()
class SegFormerYoloHeads(SgModule):
    def __init__(
        self,
        backbone,
        heads,
        num_classes: int,
    ):
        super().__init__()

        self._backbone = MiTBackBone(
            embed_dims=backbone.encoder_embed_dims,
            encoder_layers=backbone.encoder_layers,
            eff_self_att_reduction_ratio=backbone.eff_self_att_reduction_ratio,
            eff_self_att_heads=backbone.eff_self_att_heads,
            overlap_patch_size=backbone.overlap_patch_size,
            overlap_patch_stride=backbone.overlap_patch_stride,
            overlap_patch_pad=backbone.overlap_patch_pad,
            in_channels=backbone.in_channels,
        )

        factory = DetectionModulesFactory()
        heads = factory.insert_module_param(heads, "in_channels", backbone.encoder_embed_dims[1:])
        heads = factory.insert_module_param(heads, "num_classes", num_classes)

        self.heads = factory.get(heads)

    def forward(self, x):
        x = self._backbone(x)
        x = x[1:]
        return self.heads(x)


@register_model()
class SegformerB5Heads(SegFormerYoloHeads):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        recipes_dir_path = Path(__file__).parent.parent / "configs"

        default_arch_params = get_arch_params("segformer_b5_yolo", recipes_dir_path=str(recipes_dir_path))
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
        )
