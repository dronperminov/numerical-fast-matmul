import os.path
import random

import torch

from src.entities.decomposition import Decomposition
from src.entities.train_parameters import TrainParameters
from src.entities.train_strategy import TrainStrategy
from src.losses import reconstruction_loss, rationalization_loss, sparsity_loss, magnitude_loss, balance_loss
from src.utils import get_values_ring, value2str, normalize_value


class DecompositionSolver:
    def __init__(self, decomposition: Decomposition, strategy: TrainStrategy, T: torch.Tensor, output_dir: str) -> None:
        self.decomposition = decomposition
        self.strategy = strategy
        self.T = T
        self.output_dir = output_dir
        self.optimizer = self.__get_optimizer()

        self.batch_size = decomposition.batch_size
        self.device = decomposition.device
        self.best_errors = torch.full((self.batch_size,), float('inf'), dtype=torch.float64, device=self.device)
        self.verified_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

    def step(self, step: int, steps: int) -> float:
        parameters, t = self.strategy.get(step=step, steps=steps)
        loss = self.__train_step(parameters=parameters, t=t)
        self.__verify()

        if random.random() < parameters.als_probability:
            self.decomposition.als(target=self.T)
            self.__verify()

        return loss

    def __train_step(self, parameters: TrainParameters, t: float) -> float:
        self.optimizer.zero_grad()
        loss = self.__get_loss(parameters=parameters, t=t)
        loss = loss.sum()
        loss.backward()

        torch.nn.utils.clip_grad_norm_([self.decomposition.u, self.decomposition.v, self.decomposition.w], max_norm=5.0)
        self.optimizer.step()
        return loss.item() / self.batch_size

    def __verify(self) -> None:
        for scale in self.strategy.scales:
            self.__check_rationalized(scale=scale)

    def status(self) -> None:
        u, v, w = self.decomposition.u, self.decomposition.v, self.decomposition.w
        uvw_abs = torch.cat([torch.view_as_real(t) if torch.is_complex(t) else t for t in [u, v, w]], dim=1).abs()
        mask = uvw_abs > 0

        reconstruction = reconstruction_loss(target=self.T, u=u, v=v, w=w).mean().item()
        rationalization = torch.abs(uvw_abs[mask] - torch.round(uvw_abs[mask] * 2) / 2).mean().item()
        magnitude = uvw_abs[mask].mean().item()
        balance = balance_loss(u=u, v=v, w=w).mean().item()

        reconstruction_rounds = []
        for scale in self.strategy.scales:
            ur, vr, wr = self.decomposition.get_rounded(scale=scale)
            reconstruction_round = reconstruction_loss(target=self.T, u=ur, v=vr, w=wr)
            reconstruction_rounds.append(reconstruction_round)

        reconstruction_round = torch.stack(reconstruction_rounds, dim=1).min(dim=1)[0]
        round_mean = reconstruction_round.mean().item()
        round_min = reconstruction_round.min().item()
        round_best = self.best_errors.min().item()

        verified = self.verified_mask.sum().item()

        print("| reconstruction | rounded recons. (mean / min / best) | rationalization | magnitude |   balance   | verified |")
        print("+----------------+-------------------------------------+-----------------+-----------+-------------+----------+")
        print(f"| {reconstruction:14.6f} | {round_mean:11.5f} | {round_min:9.5f} | {round_best:9.5f} | {rationalization:15.6f} | {magnitude:9.6f} | {balance:11.6f} | {verified:8} |")
        print("")

    def __get_optimizer(self) -> torch.optim.Optimizer:
        if self.strategy.optimizer_name == "adam":
            return torch.optim.Adam([self.decomposition.u, self.decomposition.v, self.decomposition.w], lr=self.strategy.learning_rate)

        if self.strategy.optimizer_name == "adamw":
            return torch.optim.AdamW([self.decomposition.u, self.decomposition.v, self.decomposition.w], lr=self.strategy.learning_rate)

        raise ValueError(f'Unknown optimizer "{self.strategy.optimizer_name}"')

    def __get_loss(self, parameters: TrainParameters, t: float) -> torch.Tensor:
        w_rationalization = parameters.w_rationalization(t)
        w_sparsity = parameters.w_sparsity(t)
        w_magnitude = parameters.w_magnitude(t)
        w_balance = parameters.w_balance(t)

        u, v, w = self.decomposition.u, self.decomposition.v, self.decomposition.w
        loss = reconstruction_loss(target=self.T, u=u, v=v, w=w)

        if w_rationalization:
            loss += w_rationalization * rationalization_loss(u, v, w, rationalization_type=parameters.rationalization_type)

        if w_sparsity != 0:
            loss += w_sparsity * sparsity_loss(u, v, w, sparsity_type=parameters.sparsity_type)

        if w_magnitude != 0:
            loss += w_magnitude * magnitude_loss(u, v, w, max_abs_value=parameters.max_abs_value)

        if w_balance != 0:
            loss += w_balance * balance_loss(u, v, w)

        return loss

    def __check_rationalized(self, scale: int, eps: float = 1e-10) -> None:
        u, v, w = self.decomposition.get_rounded(scale=scale)
        errors = reconstruction_loss(target=self.T, u=u, v=v, w=w).to(self.best_errors.dtype)

        improved = errors < self.best_errors
        if not improved.any():
            return

        self.best_errors[improved] = errors[improved]
        verify_mask = (self.best_errors < eps) & (~self.verified_mask)
        self.verified_mask |= verify_mask

        if not verify_mask.any():
            return

        for index in verify_mask.nonzero(as_tuple=True)[0]:
            self.__save_verified(u[index], v[index], w[index])

    def __save_verified(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> None:
        n, m, p = self.decomposition.dimension
        elements = self.decomposition.elements
        rank = self.decomposition.rank

        complexity = sum((matrix != 0).sum().item() for matrix in [u, v, w]) - 2 * rank - n * p
        values = {normalize_value(value.item()) for matrix in [u, v, w] for row in matrix for value in row}
        ring = get_values_ring(values)

        u = [[u[i, index].item() for i in range(elements[0])] for index in range(rank)]
        v = [[v[i, index].item() for i in range(elements[1])] for index in range(rank)]
        w = [[w[i, index].item() for i in range(elements[2])] for index in range(rank)]

        u_str = "],\n        [".join([", ".join(value2str(value) for value in row) for row in u])
        v_str = "],\n        [".join([", ".join(value2str(value) for value in row) for row in v])
        w_str = "],\n        [".join([", ".join(value2str(value) for value in row) for row in w])

        filename = f"{n}x{m}x{p}_m{rank}_c{complexity}_{ring}.json"
        with open(os.path.join(self.output_dir, filename), "w") as f:
            f.write("{\n")
            f.write(f'    "dimension": [{n}, {m}, {p}],\n')
            f.write(f'    "rank": {rank},\n')
            f.write(f'    "complexity": {complexity},\n')
            f.write(f'    "ring": "{ring}",\n')
            f.write(f'    "u": [\n        [{u_str}]\n    ],\n')
            f.write(f'    "v": [\n        [{v_str}]\n    ],\n')
            f.write(f'    "w": [\n        [{w_str}]\n    ]\n')
            f.write("}\n")

        print(f"Verified {ring} scheme (complexity: {complexity}, values: {values}), total: {self.verified_mask.sum().item()}")
