from typing import Tuple

import torch


class Decomposition:
    def __init__(self, n: int, m: int, p: int, rank: int, dtype: torch.dtype, batch_size: int, device: str):
        self.dimension = [n, m, p]
        self.elements = [n * m, m * p, p * n]
        self.rank = rank

        self.dtype = dtype
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

    def copy(self) -> "Decomposition":
        n, m, p = self.dimension
        decomposition = Decomposition(n=n, m=m, p=p, rank=self.rank, dtype=self.dtype, batch_size=self.batch_size, device=self.device)

        with torch.no_grad():
            decomposition.u.data = self.u.clone()
            decomposition.v.data = self.v.clone()
            decomposition.w.data = self.w.clone()

        decomposition.u.requires_grad_(True)
        decomposition.v.requires_grad_(True)
        decomposition.w.requires_grad_(True)
        return decomposition

    def als(self, target: torch.Tensor) -> None:
        orders = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
        ]

        with torch.no_grad():
            uvw = [self.u, self.v, self.w]
            order_sequence = torch.randint(0, 6, (self.batch_size,), device=self.device)
            targets = {(axis1, axis2, axis3): target.permute(axis1, axis2, axis3).reshape(self.elements[axis1], -1) for axis1, axis2, axis3 in orders}

            for index, order in enumerate(orders):
                mask = order_sequence == index
                if not mask.any():
                    continue

                for i, axis1 in enumerate(order):
                    axis2, axis3 = order[(i + 1) % 3], order[(i + 2) % 3]
                    uvw[axis1][mask] = self.__als_step(uvw[axis2][mask], uvw[axis3][mask], targets[(axis1, axis2, axis3)])

            self.u.data = uvw[0]
            self.v.data = uvw[1]
            self.w.data = uvw[2]

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

    def __als_step(self, v: torch.Tensor, w: torch.Tensor, T: torch.Tensor, lambda_reg: float = 1e-15) -> torch.Tensor:
        batch_size = v.shape[0]
        vw = torch.einsum('bik,bjk->bijk', v, w).reshape(batch_size, -1, self.rank)
        a = torch.einsum('bri,brj->bij', vw.conj(), vw)
        b = torch.einsum('ij,bjk->bik', T.to(self.dtype), vw)
        eye = torch.eye(self.rank, dtype=self.dtype, device=self.device).unsqueeze(0)
        u = torch.linalg.solve(a + lambda_reg * eye, b.permute(0, 2, 1).conj())
        return u.permute(0, 2, 1)
