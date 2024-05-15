import torch
from torch import nn, Tensor


class FrobeniusNormLoss(nn.Module):
    def __call__(self, R1: Tensor, R2: Tensor):
        return torch.norm(R1 - R2, p='fro', dim=(1, 2)).mean()


class GeodesicLoss(nn.Module):
    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        R_diffs = input @ target.permute(0, 2, 1)
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()


class CosineRotationLoss(nn.Module):
    def __call__(self, R1: Tensor, R2: Tensor):
        product = torch.matmul(R1.transpose(1, 2), R2)
        trace = product.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        similarity = trace / 3.0
        loss = 1 - similarity
        return loss.mean()
