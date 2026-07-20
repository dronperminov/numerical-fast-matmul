import random

import torch


def reconstruction_loss(target: torch.Tensor, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    reconstructed = torch.einsum("bir,bjr,bkr->bijk", u, v, w)
    residual = target.to(u.dtype).unsqueeze(0).expand_as(reconstructed) - reconstructed
    loss = torch.real(residual * residual.conj()) if torch.is_complex(u) else residual ** 2
    return loss.flatten(start_dim=1).sum(dim=1)


def rationalization_loss(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, rationalization_type: str) -> torch.Tensor:
    if rationalization_type == "random":
        rationalization_type = random.choice(["sin", "round", "ternary"])

    if rationalization_type == "sin":
        return rationalization_loss_sin(u=u, v=v, w=w)

    if rationalization_type == "round":
        return rationalization_loss_round(u=u, v=v, w=w)

    if rationalization_type == "ternary":
        return rationalization_loss_ternary(u=u, v=v, w=w)

    raise ValueError(f'Unknown rationalization type "{rationalization_type}"')


def sparsity_loss(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, sparsity_type: str) -> torch.Tensor:
    if sparsity_type == "random":
        sparsity_type = random.choice(["sqrt", "l1"])

    if sparsity_type == "sqrt":
        return sparsity_loss_sqrt(u=u, v=v, w=w)

    if sparsity_type == "l1":
        return sparsity_loss_l1(u=u, v=v, w=w)

    raise ValueError(f'Unknown sparsity type "{sparsity_type}"')


def magnitude_loss(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, max_abs_value: float) -> torch.Tensor:
    loss = 0.0
    for matrix in [u, v, w]:
        if torch.is_complex(matrix):
            matrix = torch.view_as_real(matrix)

        loss += (torch.relu(matrix.abs() - max_abs_value) ** 2).flatten(start_dim=1).sum(dim=1)

    return loss


def balance_loss(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    norm_u = (torch.abs(u) ** 2).sum(dim=1)
    norm_v = (torch.abs(v) ** 2).sum(dim=1)
    norm_w = (torch.abs(w) ** 2).sum(dim=1)
    norm_mean = (norm_u * norm_v * norm_w).clamp(min=1e-8) ** (1 / 3)

    loss = 0.0
    for norm in [norm_u, norm_v, norm_w]:
        loss += ((norm / norm_mean - 1.0) ** 2).sum(dim=1)

    return loss


def rationalization_loss_sin(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for matrix in [u, v, w]:
        if torch.is_complex(matrix):
            matrix = torch.view_as_real(matrix)
        loss += torch.abs(torch.sin(torch.pi * matrix)).flatten(start_dim=1).sum(dim=1)

    return loss


def rationalization_loss_round(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    loss = 0.0

    for matrix in [u, v, w]:
        if torch.is_complex(matrix):
            matrix = torch.view_as_real(matrix)

        for scale in [1, 2]:
            loss += torch.abs(matrix - torch.round(matrix * scale) / scale).flatten(start_dim=1).sum(dim=1)

    return loss


def rationalization_loss_ternary(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for matrix in [u, v, w]:
        if torch.is_complex(matrix):
            matrix = torch.view_as_real(matrix)

        w = torch.abs(matrix)
        wp = torch.abs(matrix - 1)
        wn = torch.abs(matrix + 1)
        loss += (w*wp*wn).flatten(start_dim=1).sum(dim=1)

    return loss


def sparsity_loss_sqrt(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    loss = 0.0
    for matrix in [u, v, w]:
        if torch.is_complex(matrix):
            matrix = torch.view_as_real(matrix)

        loss += (torch.sqrt(matrix ** 2 + eps ** 2) - eps).flatten(start_dim=1).sum(dim=1)

    return loss


def sparsity_loss_l1(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for matrix in [u, v, w]:
        if torch.is_complex(matrix):
            matrix = torch.view_as_real(matrix)

        loss += torch.abs(matrix).flatten(start_dim=1).sum(dim=1)

    return loss
