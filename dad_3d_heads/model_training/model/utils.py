from typing import Optional, List, Union
from smplx.utils import Struct
from dad_3d_heads.utils import get_relative_path
import pickle

import torch
import torch.nn.functional as F


def to_device(
    x: Union[torch.Tensor, torch.nn.Module, torch.ScriptModule], cuda_id: int = 0
) -> Union[torch.Tensor, torch.nn.Module, torch.ScriptModule]:
    return x.cuda(cuda_id) if torch.cuda.is_available() else x


def unravel_index(x: torch.Tensor) -> torch.Tensor:
    """Extract maximum value along the first axis.
    Used to extract keypoints coordinates from the heatmap
    Args:
       x (torch.Tensor): Image with shape :math:`(B, C, H, W)`.
    Returns:
       torch.Tensor: Max coordinates with shape :math:`(B, C, 2)`.
    Example:
        >>> tensor = torch.rand(2, 68, 64, 64)
        >>> keypoints = unravel_index(tensor)
    """
    B, C, H, W = x.shape
    max_tensor = x.view(B, C, -1).argmax(-1).view(-1, 1)
    keypoints = torch.cat((torch.div(max_tensor, H, rounding_mode="trunc"), max_tensor % H), dim=1).reshape(B, C, 2)
    return keypoints


def normalize_to_cube(v: torch.Tensor) -> torch.Tensor:
    """
    Normalizes vertices (of a mesh) to the unit cube [-1;1]^3.
    Args:
        v: [B, N, 3] - vertices.
        Handles [N, 3] vertices as well.
    Returns:
        (modified) v: [B, N, 3].
    """
    if v.ndim == 2:
        v = v[None]
    v = v - v.min(1, True)[0]
    v = v - 0.5 * v.max(1, True)[0]
    return v / v.max(-1, True)[0].max(-2, True)[0]


def calculate_paddings(orig_h: int, orig_w: int) -> List[int]:
    max_orig_side = max(orig_h, orig_w)
    pad_top = int((max_orig_side - orig_h) / 2)
    pad_bottom = max_orig_side - orig_h - pad_top
    pad_left = int((max_orig_side - orig_w) / 2)
    pad_right = max_orig_side - orig_w - pad_left
    return [pad_top, pad_bottom, pad_left, pad_right]


def get_flame_model(flame_path: Optional[str] = None) -> Struct:
    if flame_path is None:
        flame_path = get_relative_path(f"../../model_3d/generic_model.pkl", __file__)

    with open(flame_path, "rb") as f:
        flame_model = Struct(**pickle.load(f, encoding="latin1"))
    return flame_model


def rot_mat_from_6dof(v: torch.Tensor) -> torch.Tensor:
    assert v.shape[-1] == 6
    v = v.view(-1, 6)
    vx, vy = v[..., :3].clone(), v[..., 3:].clone()

    b1 = F.normalize(vx, dim=-1)
    b3 = F.normalize(torch.cross(b1, vy), dim=-1)
    b2 = - torch.cross(b1, b3)

    return torch.stack((b1, b2, b3), dim=-1)