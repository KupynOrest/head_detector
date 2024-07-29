import os.path
import pickle
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import einops
import numpy as np
import torch.nn as nn
from smplx.lbs import lbs
from smplx.utils import to_tensor, to_np
from smplx.utils import Struct
import torch
from torch import Tensor

from head_detector.utils import rot_mat_from_6dof


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


def get_flame_model(flame_path: Optional[str] = None) -> Struct:
    if flame_path is None:
        flame_path = os.path.join(os.path.dirname(__file__), f"generic_model.pkl")

    with open(flame_path, "rb") as f:
        flame_model = Struct(**pickle.load(f, encoding="latin1"))
    return flame_model


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


MAX_SHAPE = 300
MAX_EXPRESSION = 100

ROT_COEFFS = 3
JAW_COEFFS = 3
EYE_COEFFS = 6
NECK_COEFFS = 3
MESH_OFFSET_Z = 0.05


class FLAMELayer(nn.Module):
    """
    Based on https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function which outputs vertices of the FLAME mesh, modified w.r.t. these parameters.
    """

    def __init__(self, consts: Dict[str, Any] = None, batch_size: int = 1, flame_path: Optional[str] = None) -> None:
        super().__init__()
        if consts is None:
            consts = FLAME_CONSTS
        self.flame_model = get_flame_model(flame_path)
        self.flame_constants = consts
        self.batch_size = batch_size
        self.dtype = torch.float32
        self.faces = self.flame_model.f
        self.register_buffer("faces_tensor", to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long))
        # Fixing remaining Shape betas
        default_shape = torch.zeros([self.batch_size, MAX_SHAPE - consts["shape"]], dtype=self.dtype, requires_grad=False)
        self.register_parameter("shape_betas", nn.Parameter(default_shape, requires_grad=False))

        # Fixing remaining expression betas
        default_exp = torch.zeros([self.batch_size, MAX_EXPRESSION - consts["expression"]], dtype=self.dtype, requires_grad=False)
        self.register_parameter("expression_betas", nn.Parameter(default_exp, requires_grad=False))

        default_rot = torch.zeros([self.batch_size, ROT_COEFFS], dtype=self.dtype, requires_grad=False)
        self.register_parameter("rot", nn.Parameter(default_rot, requires_grad=False))

        default_jaw = torch.zeros([self.batch_size, JAW_COEFFS], dtype=self.dtype, requires_grad=False)
        self.register_parameter("jaw", nn.Parameter(default_jaw, requires_grad=False))

        # Eyeball and neck rotation
        default_eyeball_pose = torch.zeros([self.batch_size, EYE_COEFFS], dtype=self.dtype, requires_grad=False)
        self.register_parameter("eyeballs", nn.Parameter(default_eyeball_pose, requires_grad=False))

        default_neck_pose = torch.zeros([self.batch_size, NECK_COEFFS], dtype=self.dtype, requires_grad=False)
        self.register_parameter("neck_pose", nn.Parameter(default_neck_pose, requires_grad=False))

        # The vertices of the template model
        self.register_buffer("v_template", to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer("shapedirs", to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer("J_regressor", j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        self.register_buffer("lbs_weights", to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))

    def forward_spatial(self, flame_params: FlameParams, zero_rot: bool = False, zero_jaw: bool = False):
        """
        forward() version that takes dense feature map of flame parameters as input [B,F,A]
        Args:
            flame_params:
            zero_rot:
            zero_jaw:

        Returns:

        """
        b, _, a = flame_params.shape.size()
        new_flame_params = FlameParams(
            shape=einops.rearrange(flame_params.shape, "b c a -> (b a) c"),
            expression=einops.rearrange(flame_params.expression, "b c a -> (b a) c"),
            rotation=einops.rearrange(flame_params.rotation, "b c a -> (b a) c"),
            translation=einops.rearrange(flame_params.translation, "b c a -> (b a) c"),
            scale=einops.rearrange(flame_params.scale, "b c a -> (b a) c"),
            jaw=einops.rearrange(flame_params.jaw, "b c a -> (b a) c"),
            eyeballs=einops.rearrange(flame_params.eyeballs, "b c a -> (b a) c"),
            neck=einops.rearrange(flame_params.neck, "b c a -> (b a) c"),
        )
        vertices = self.forward(new_flame_params, zero_rot, zero_jaw)
        return einops.rearrange(vertices, "(b a) v c -> b a v c", a=a)

    def forward(self, flame_params: FlameParams, zero_rot: bool = False, zero_jaw: bool = False) -> torch.Tensor:
        """
        Input:
            shape_params: B X number of shape parameters
            expression_params: B X number of expression parameters
            pose_params: B X number of pose parameters
        return:
            vertices: B X V X 3
        """
        bs = flame_params.shape.shape[0]
        betas = torch.cat(
            [
                flame_params.shape,
                self.shape_betas[[0]].expand(bs, -1),
                flame_params.expression,
                self.expression_betas[[0]].expand(bs, -1),
            ],
            dim=1,
        )
        neck_pose = flame_params.neck if not (0 in flame_params.neck.shape) else self.neck_pose[[0]].expand(bs, -1)
        eyeballs = flame_params.eyeballs if not (0 in flame_params.eyeballs.shape) else self.eyeballs[[0]].expand(bs, -1)
        jaw = flame_params.jaw if not (0 in flame_params.jaw.shape) else self.jaw[[0]].expand(bs, -1)

        rotation = torch.zeros([bs, ROT_COEFFS], device=flame_params.rotation.device)
        if zero_jaw:
            jaw = torch.zeros_like(jaw)
        full_pose = torch.cat([rotation, neck_pose, jaw, eyeballs], dim=1)

        template_vertices = self.v_template.unsqueeze(0).repeat(bs, 1, 1)

        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )

        # translate to skull center and rotate
        vertices[:, :, 2] += MESH_OFFSET_Z
        if not zero_rot:
            rotation_mat = rot_mat_from_6dof(flame_params.rotation).type(vertices.dtype)
            vertices = torch.matmul(rotation_mat.unsqueeze(1), vertices.unsqueeze(-1))
            vertices = vertices[..., 0]
        return vertices


def uint8_to_float32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        return x.div(255.0).to(dtype=torch.float32)
    else:
        return x


def reproject_spatial_vertices(flame: FLAMELayer, flame_params: Tensor, to_2d: bool = True, subset_indexes=None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    :param flame_params: [..., Num Flame Params]
    :return: [..., Num Vertices, 2] if to_2d else [..., Num Vertices, 3]
    """
    shape = flame_params.size()
    # If there are no flame parameters, return zeros
    if flame_params.size(0) == 0:
        projected_vertices = torch.zeros((0,) + (flame.v_template.size(0), 2 if to_2d else 3), device=flame_params.device)
        vertices = torch.zeros((0,) + (flame.v_template.size(0), 3), device=flame_params.device)
        rotation_mat = torch.eye(3, device=flame_params.device).unsqueeze(0).expand(flame_params.size(0), 3, 3)
    else:
        flame_params_inp = FlameParams.from_3dmm(flame_params, FLAME_CONSTS)
        vertices = flame.forward(flame_params_inp, zero_rot=True)
        # translate to skull center and rotate
        rot_vertices = vertices.clone()
        rotation_mat = rot_mat_from_6dof(flame_params_inp.rotation).type(vertices.dtype)
        rot_vertices = torch.matmul(rotation_mat.unsqueeze(1), rot_vertices.unsqueeze(-1))
        rot_vertices = rot_vertices[..., 0]
        scale = torch.clamp(flame_params_inp.scale[:, None], 1e-8)
        projected_vertices = (rot_vertices * scale) + flame_params_inp.translation[:, None]  # [B, 1, 3]

    if subset_indexes is not None:
        projected_vertices = projected_vertices[:, subset_indexes]
    if to_2d:
        projected_vertices = projected_vertices[..., :2]

    # Reshape back to the original shape
    projected_vertices = projected_vertices.view(*shape[:-1], *projected_vertices.size()[-2:]).contiguous()
    return vertices, rotation_mat, projected_vertices
