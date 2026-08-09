from typing import List, Tuple

import torch

from src.decompositions.abstract_decomposition import AbstractDecomposition


class CyclicDecomposition(AbstractDecomposition):
    def __init__(self, n: int, s: int, t: int, rank: int, dtype: torch.dtype, batch_size: int, device: str):
        super().__init__(n=n, m=n, p=n, rank=rank, dtype=dtype, batch_size=batch_size, device=device)

        self.s = s
        self.t = t
        self.r = rank - s - 3 * t
        assert rank >= s + 3 * t

        self.a = torch.zeros(self.batch_size, n * n, self.s, device=self.device, dtype=self.dtype, requires_grad=True)
        self.b = torch.zeros(self.batch_size, n * n, self.t, device=self.device, dtype=self.dtype, requires_grad=True)
        self.c = torch.zeros(self.batch_size, n * n, self.t, device=self.device, dtype=self.dtype, requires_grad=True)
        self.d = torch.zeros(self.batch_size, n * n, self.t, device=self.device, dtype=self.dtype, requires_grad=True)

        self.u = torch.zeros(self.batch_size, n * n, self.r, device=self.device, dtype=self.dtype, requires_grad=True)
        self.v = torch.zeros(self.batch_size, n * n, self.r, device=self.device, dtype=self.dtype, requires_grad=True)
        self.w = torch.zeros(self.batch_size, n * n, self.r, device=self.device, dtype=self.dtype, requires_grad=True)

    def get_parameters(self) -> List[torch.Tensor]:
        return [self.a, self.b, self.c, self.d, self.u, self.v, self.w]

    def get_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = torch.cat((self.a, self.b, self.c, self.d, self.u), dim=2)
        v = torch.cat((self.a, self.c, self.d, self.b, self.v), dim=2)
        w = torch.cat((self.a, self.d, self.b, self.c, self.w), dim=2)
        return u, v, w

    def get_rounded(self, scale: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            a = self._round(self.a, scale=scale)
            b = self._round(self.b, scale=scale)
            c = self._round(self.c, scale=scale)
            d = self._round(self.d, scale=scale)

            u = self._round(self.u, scale=scale)
            v = self._round(self.v, scale=scale)
            w = self._round(self.w, scale=scale)

        u = torch.cat((a, b, c, d, u), dim=2)
        v = torch.cat((a, c, d, b, v), dim=2)
        w = torch.cat((a, d, b, c, w), dim=2)
        return u, v, w

    def initialize(self, scale: float = 0.5) -> None:
        with torch.no_grad():
            self.a.normal_(mean=0, std=scale)
            self.b.normal_(mean=0, std=scale)
            self.c.normal_(mean=0, std=scale)
            self.d.normal_(mean=0, std=scale)

            self.u.normal_(mean=0, std=scale)
            self.v.normal_(mean=0, std=scale)
            self.w.normal_(mean=0, std=scale)

    def copy(self) -> "AbstractDecomposition":
        n, n, n = self.dimension
        decomposition = CyclicDecomposition(n=n, s=self.s, t=self.t, rank=self.rank, dtype=self.dtype, batch_size=self.batch_size, device=self.device)

        with torch.no_grad():
            decomposition.a.copy_(self.a)
            decomposition.b.copy_(self.b)
            decomposition.c.copy_(self.c)
            decomposition.d.copy_(self.d)

            decomposition.u.copy_(self.u)
            decomposition.v.copy_(self.v)
            decomposition.w.copy_(self.w)

        return decomposition

    def als(self, target: torch.Tensor) -> None:
        u, v, w = self._als(target)

        with torch.no_grad():
            a_u = u[:, :, :self.s]
            b_u = u[:, :, self.s:self.s + self.t]
            c_u = u[:, :, self.s + self.t:self.s + 2 * self.t]
            d_u = u[:, :, self.s + 2 * self.t:self.s + 3 * self.t]

            a_v = v[:, :, :self.s]
            c_v = v[:, :, self.s:self.s + self.t]
            d_v = v[:, :, self.s + self.t:self.s + 2 * self.t]
            b_v = v[:, :, self.s + 2 * self.t:self.s + 3 * self.t]

            a_w = w[:, :, :self.s]
            d_w = w[:, :, self.s:self.s + self.t]
            b_w = w[:, :, self.s + self.t:self.s + 2 * self.t]
            c_w = w[:, :, self.s + 2 * self.t:self.s + 3 * self.t]

            self.a.copy_((a_u + a_v + a_w) / 3.0).clamp_(-2.0, 2.0)
            self.b.copy_((b_u + b_v + b_w) / 3.0).clamp_(-2.0, 2.0)
            self.c.copy_((c_u + c_v + c_w) / 3.0).clamp_(-2.0, 2.0)
            self.d.copy_((d_u + d_v + d_w) / 3.0).clamp_(-2.0, 2.0)

            self.u.copy_(u[:, :, self.s + 3 * self.t:])
            self.v.copy_(v[:, :, self.s + 3 * self.t:])
            self.w.copy_(w[:, :, self.s + 3 * self.t:])
