import torch
from torch import nn, Tensor


__all__ = ["Vertices3DLoss"]

losses = {"l1": nn.L1Loss, "l2": nn.MSELoss, "smooth_l1": nn.SmoothL1Loss}


def normalize_to_cube(v: Tensor) -> Tensor:
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


class Vertices3DLoss(nn.Module):
    def __init__(
        self,
        criterion,
    ):
        super().__init__()
        if criterion not in losses.keys():
            raise ValueError(f"Unsupported discrepancy loss type {criterion}")
        self.criterion = losses[criterion]()

    @torch.cuda.amp.autocast(False)
    def forward(self, predicted: Tensor, target: Tensor) -> Tensor:
        """

        Args:
            predicted: [B,N,3]
            target: [B,N,3]

        Returns:

        """

        loss = self.criterion(*tuple(map(normalize_to_cube, (predicted, target))))
        return loss
