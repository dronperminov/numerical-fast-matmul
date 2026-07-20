from typing import Tuple

import torch


class Decomposition:
    def __init__(self, n: int, m: int, p: int, rank: int, data_type: str, batch_size: int, device: str):
        self.dimension = [n, m, p]
        self.elements = [n * m, m * p, p * n]
        self.rank = rank

        data_type2dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
            "complex64": torch.complex64,
            "complex128": torch.complex128
        }
        self.dtype = data_type2dtype[data_type]
        self.batch_size = batch_size
        self.device = device

        self.u = torch.zeros(self.batch_size, self.elements[0], self.rank, device=self.device, dtype=self.dtype)
        self.v = torch.zeros(self.batch_size, self.elements[1], self.rank, device=self.device, dtype=self.dtype)
        self.w = torch.zeros(self.batch_size, self.elements[2], self.rank, device=self.device, dtype=self.dtype)

    def initialize(self, scale: float = 0.01) -> None:
        with torch.no_grad():
            self.u.data = torch.randn_like(self.u) * scale
            self.v.data = torch.randn_like(self.v) * scale
            self.w.data = torch.randn_like(self.w) * scale

        self.u.requires_grad_(True)
        self.v.requires_grad_(True)
        self.w.requires_grad_(True)

    def get_rounded(self, scale: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            u = self.__round(self.u, scale=scale)
            v = self.__round(self.v, scale=scale)
            w = self.__round(self.w, scale=scale)

        return u, v, w

    def __round(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        if torch.is_complex(x):
            x = torch.view_as_real(x)
            x = torch.round(x * scale) / scale
            return torch.view_as_complex(x)

        return torch.round(x * scale) / scale
