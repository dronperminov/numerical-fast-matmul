from abc import abstractmethod, ABC
from typing import List, Tuple

import torch


class AbstractDecomposition(ABC):
    def __init__(self, n: int, m: int, p: int, rank: int, dtype: torch.dtype, batch_size: int, device: str) -> None:
        self.dimension = [n, m, p]
        self.elements = [n * m, m * p, p * n]
        self.rank = rank

        self.dtype = dtype
        self.batch_size = batch_size
        self.device = device

    def get_coefficients_count(self) -> int:
        return self.rank * sum(self.elements)

    @abstractmethod
    def get_parameters(self) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def get_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def get_rounded(self, scale: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def initialize(self, scale: float = 0.5) -> None:
        pass

    @abstractmethod
    def copy(self) -> "AbstractDecomposition":
        pass

    @abstractmethod
    def als(self, target: torch.Tensor) -> None:
        pass

    def project_to_rounded(self, scale: int, alpha: float):
        with torch.no_grad():
            for matrix in self.get_parameters():
                matrix.copy_(self.__project_round(matrix, scale=scale, alpha=alpha))

    def _als(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orders = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
        ]

        with torch.no_grad():
            uvw = [matrix.clone() for matrix in self.get_coefficients()]
            order_sequence = torch.randint(0, 6, (self.batch_size,), device=self.device)
            targets = {(axis1, axis2, axis3): target.permute(axis1, axis2, axis3).reshape(self.elements[axis1], -1) for axis1, axis2, axis3 in orders}

            for index, order in enumerate(orders):
                mask = order_sequence == index
                if not mask.any():
                    continue

                for i, axis1 in enumerate(order):
                    axis2, axis3 = order[(i + 1) % 3], order[(i + 2) % 3]
                    uvw[axis1][mask] = self.__als_step(uvw[axis2][mask], uvw[axis3][mask], targets[(axis1, axis2, axis3)])

        return uvw[0], uvw[1], uvw[2]

    def _round(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        if torch.is_complex(x):
            x = torch.view_as_real(x)
            x = torch.round(x * scale) / scale
            return torch.view_as_complex(x)

        return torch.round(x * scale) / scale

    def __project_round(self, x: torch.Tensor, scale: int, alpha: float)-> torch.Tensor:
        if torch.is_complex(x):
            x = torch.view_as_real(x)
            x = (1 - alpha) * x + alpha * torch.round(x * scale) / scale
            return torch.view_as_complex(x)

        return (1 - alpha) * x + alpha * torch.round(x * scale) / scale

    def __als_step(self, v: torch.Tensor, w: torch.Tensor, T: torch.Tensor, lambda_reg: float = 1e-15) -> torch.Tensor:
        batch_size = v.shape[0]
        vw = torch.einsum('bik,bjk->bijk', v, w).reshape(batch_size, -1, self.rank)
        a = torch.einsum('bri,brj->bij', vw.conj(), vw)
        b = torch.einsum('ij,bjk->bik', T, vw)
        eye = torch.eye(self.rank, dtype=self.dtype, device=self.device).unsqueeze(0)
        u = torch.linalg.solve(a + lambda_reg * eye, b.permute(0, 2, 1).conj())
        return u.permute(0, 2, 1)
