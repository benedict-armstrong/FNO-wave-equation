import torch


def relative_l2_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(output - target) / torch.norm(target)
