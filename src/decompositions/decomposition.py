from typing import List, Tuple

import torch

from src.decompositions.abstract_decomposition import AbstractDecomposition


class Decomposition(AbstractDecomposition):
    def __init__(self, n: int, m: int, p: int, rank: int, dtype: torch.dtype, batch_size: int, device: str) -> None:
        super().__init__(n=n, m=m, p=p, rank=rank, dtype=dtype, batch_size=batch_size, device=device)

        self.u = torch.zeros(self.batch_size, self.elements[0], self.rank, device=self.device, dtype=self.dtype, requires_grad=True)
        self.v = torch.zeros(self.batch_size, self.elements[1], self.rank, device=self.device, dtype=self.dtype, requires_grad=True)
        self.w = torch.zeros(self.batch_size, self.elements[2], self.rank, device=self.device, dtype=self.dtype, requires_grad=True)

    def get_parameters(self) -> List[torch.Tensor]:
        return [self.u, self.v, self.w]

    def get_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.u, self.v, self.w

    def get_rounded(self, scale: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            u = self._round(self.u, scale=scale)
            v = self._round(self.v, scale=scale)
            w = self._round(self.w, scale=scale)

        return u, v, w

    def initialize(self, scale: float = 0.5) -> None:
        with torch.no_grad():
            self.u.normal_(mean=0, std=scale)
            self.v.normal_(mean=0, std=scale)
            self.w.normal_(mean=0, std=scale)

    def copy(self) -> "AbstractDecomposition":
        n, m, p = self.dimension
        decomposition = Decomposition(n=n, m=m, p=p, rank=self.rank, dtype=self.dtype, batch_size=self.batch_size, device=self.device)

        with torch.no_grad():
            decomposition.u.copy_(self.u)
            decomposition.v.copy_(self.v)
            decomposition.w.copy_(self.w)

        return decomposition

    def als(self, target: torch.Tensor) -> None:
        u, v, w = self._als(target)

        with torch.no_grad():
            self.u.copy_(u)
            self.v.copy_(v)
            self.w.copy_(w)
