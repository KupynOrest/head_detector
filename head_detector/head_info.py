from dataclasses import dataclass
from collections import namedtuple
from typing import Optional, Dict

import torch
from torch import Tensor
import numpy as np

Bbox = namedtuple("Bbox", ["x", "y", "w", "h"])
RPY = namedtuple("RPY", ["roll", "pitch", "yaw"])

FLAME_CONSTS = {
    "shape": 300,
    "expression": 100,
    "rotation": 6,
    "jaw": 3,
    "eyeballs": 0,
    "neck": 0,
    "translation": 3,
    "scale": 1,
}


@dataclass
class HeadMetadata:
    bbox: Bbox
    score: float
    flame_params: object
    vertices_3d: np.ndarray
    head_pose: RPY


@dataclass
class FlameParams:
    shape: Tensor
    expression: Tensor
    rotation: Tensor
    translation: Tensor
    scale: Tensor
    jaw: Tensor
    eyeballs: Tensor
    neck: Tensor

    @classmethod
    def from_3dmm(cls, tensor_3dmm: Tensor, constants: Optional[Dict[str, int]] = None, zero_expr: bool = False) -> "FlameParams":
        """
        tensor_3dmm: [B, num_params, ...]
        """
        if constants is None:
            constants = FLAME_CONSTS
        if tensor_3dmm.size(1) != sum(constants.values()):
            raise ValueError(f"Invalid number of parameters. Expected: {sum(constants.values())}. Got: {tensor_3dmm.size(1)}.")
        cur_index: int = 0
        shape = tensor_3dmm[:, 0 : constants["shape"]]
        cur_index += constants["shape"]

        expression = tensor_3dmm[:, cur_index : cur_index + constants["expression"]]
        if zero_expr:
            expression = torch.zeros_like(expression)
        cur_index += constants["expression"]

        jaw = tensor_3dmm[:, cur_index : cur_index + constants["jaw"]]
        cur_index += constants["jaw"]

        rotation = tensor_3dmm[:, cur_index : cur_index + constants["rotation"]]
        cur_index += constants["rotation"]

        eyeballs = tensor_3dmm[:, cur_index : cur_index + constants["eyeballs"]]
        cur_index += constants["eyeballs"]

        neck = tensor_3dmm[:, cur_index : cur_index + constants["neck"]]
        cur_index += constants["neck"]

        translation = tensor_3dmm[:, cur_index : cur_index + constants["translation"]]
        cur_index += constants["translation"]

        scale = tensor_3dmm[:, cur_index : cur_index + constants["scale"]]
        cur_index += constants["scale"]

        return FlameParams(
            shape=shape,
            expression=expression,
            rotation=rotation,
            jaw=jaw,
            eyeballs=eyeballs,
            neck=neck,
            translation=translation,
            scale=scale,
        )

    def to_3dmm_tensor(self) -> Tensor:
        """
        Returns: [B,C,...]
        """
        params_3dmm = torch.cat(
            [
                self.shape,
                self.expression,
                self.rotation,
                self.jaw,
                self.eyeballs,
                self.neck,
                self.translation,
                self.scale,
            ],
            dim=1,
        )

        return params_3dmm

