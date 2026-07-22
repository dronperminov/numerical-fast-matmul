from typing import List

import torch

from src.decomposition_solver import DecompositionSolver
from src.entities.decomposition import Decomposition
from src.entities.train_strategy import TrainStrategy


class StrategyComparator:
    def __init__(self, decomposition: Decomposition, strategies: List[TrainStrategy], T: torch.Tensor, output_dir: str) -> None:
        self.decomposition = decomposition
        self.strategies = strategies
        self.T = T
        self.output_dir = output_dir

        self.solvers = []
        self.label_len = max(8, max(len(strategy.label) for strategy in strategies))
        self.verified = {strategy.label: 0 for strategy in self.strategies}
        self.verified_total = {strategy.label: 0 for strategy in self.strategies}

    def run(self, epoches: int, steps: int, log_period: int, print_verified: bool) -> None:
        for epoch in range(epoches):
            for step in range(steps):
                self.__step(step, steps, print_verified=print_verified)

                if step % log_period == 0 or step == steps - 1:
                    self.__status(epoch, step, steps)

        for solver in self.solvers:
            self.verified_total[solver.strategy.label] += solver.verified_count()

    def reset(self) -> None:
        self.decomposition.initialize()
        self.solvers = [DecompositionSolver(self.decomposition.copy(), strategy, self.T, self.output_dir) for strategy in self.strategies]

    def __step(self, step: int, steps: int, print_verified: bool = True) -> None:
        for solver in self.solvers:
            solver.step(step=step, steps=steps, print_verified=print_verified)

    def __status(self, epoch: int, step: int, steps: int) -> None:
        n, m, p = self.decomposition.dimension
        print(f"\n({n}, {m}, {p}: {self.decomposition.rank}): epoch {epoch + 1}, step {step} / {steps}")

        self.__update_verified_count()

        print(f'| {"strategy":{self.label_len}} | reconstruction | rounded recons. (mean / min / best) | rationalization | magnitude |   balance   | verified |')
        print(f'+-{"-" * self.label_len}-+----------------+-------------------------------------+-----------------+-----------+-------------+----------+')

        for solver in sorted(self.solvers, key=lambda s: (-self.verified[s.strategy.label], s.strategy.label, s.verified_count())):
            print(f"| {solver.strategy.label:{self.label_len}} ", end="")
            solver.status()

        print(f'+-{"-" * self.label_len}-+----------------+-------------------------------------+-----------------+-----------+-------------+----------+')
        if sum(self.verified.values()) + sum(self.verified_total.values()) == 0:
            return

        for label, count in sorted(self.verified.items(), key=lambda item: (-item[1], item[0])):
            print(f"{label}: {count} schemes (total: {count + self.verified_total[label]})")

    def __update_verified_count(self) -> None:
        for strategy in self.strategies:
            self.verified[strategy.label] = 0

        for solver in self.solvers:
            self.verified[solver.strategy.label] += solver.verified_count()
