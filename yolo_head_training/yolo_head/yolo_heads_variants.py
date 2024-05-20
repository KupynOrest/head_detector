import copy
from pathlib import Path
from typing import Union

from omegaconf import DictConfig
from super_gradients.common.registry import register_model
from super_gradients.training.models import get_arch_params
from super_gradients.training.utils import HpmStruct, get_param

from yolo_head.yolo_heads import YoloHeads


@register_model()
class YoloHeads_M(YoloHeads):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        recipes_dir_path = Path(__file__).parent.parent / "configs"

        default_arch_params = get_arch_params("yolo_heads_m_arch_params", recipes_dir_path=str(recipes_dir_path))
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model()
class YoloHeads_L(YoloHeads):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        recipes_dir_path = Path(__file__).parent.parent / "configs"

        default_arch_params = get_arch_params("yolo_heads_l_arch_params", recipes_dir_path=str(recipes_dir_path))
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )

    @property
    def num_classes(self):
        return self.heads.num_classes
