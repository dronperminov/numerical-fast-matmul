import time
from typing import Dict, List

import torch

from src.decomposition_solver import DecompositionSolver
from src.decompositions import AbstractDecomposition
from src.entities.train_strategy import TrainStrategy
from src.utils import pretty_time


class StrategyComparator:
    def __init__(self, decomposition: AbstractDecomposition, strategies: List[TrainStrategy], T: torch.Tensor, output_dir: str) -> None:
        self.decomposition = decomposition
        self.strategies = strategies
        self.T = T
        self.output_dir = output_dir

        self.solvers = []
        self.label_len = max(8, max(len(strategy.label) for strategy in strategies))
        self.verified_total = {strategy.label: 0 for strategy in self.strategies}
        self.label2count = self.__get_labels_count()
        self.runs = 0

    def run(self, epochs: int, steps: int, log_period: int, print_verified: bool) -> None:
        self.decomposition.initialize()
        self.solvers = [DecompositionSolver(self.decomposition.copy(), strategy, self.T, self.output_dir) for strategy in self.strategies]
        self.runs += 1
        self.__status(0, -1, steps)
        start_time = time.time()

        for epoch in range(epochs):
            for step in range(steps):
                self.__step(step, steps, print_verified=print_verified)

                if (step + 1) % log_period == 0 or step == steps - 1:
                    end_time = time.time()
                    self.__status(epoch, step, steps, elapsed=end_time - start_time)
                    start_time = end_time

        for solver in self.solvers:
            self.verified_total[solver.strategy.label] += solver.verified_count()

    def __step(self, step: int, steps: int, print_verified: bool = True) -> None:
        for solver in self.solvers:
            solver.step(step=step, steps=steps, print_verified=print_verified)

    def __status(self, epoch: int, step: int, steps: int, elapsed: float = 0.0) -> None:
        n, m, p = self.decomposition.dimension
        print(f"\n({n}, {m}, {p}: {self.decomposition.rank}): run: {self.runs}, epoch {epoch + 1}, step {step + 1} / {steps}, elapsed: {pretty_time(elapsed)}")
        print(f'| {"strategy":{self.label_len}} | reconstruction | rounded recons. (mean / min / best) | rationalization | magnitude |   balance   | verified |')
        print(f'+-{"-" * self.label_len}-+----------------+-------------------------------------+-----------------+-----------+-------------+----------+')

        verified = self.__get_verified_count()
        for solver in sorted(self.solvers, key=lambda s: (-verified[s.strategy.label], s.strategy.label, -s.verified_count())):
            print(f"| {solver.strategy.label:{self.label_len}} ", end="")
            solver.status()

        print(f'+-{"-" * self.label_len}-+----------------+-------------------------------------+-----------------+-----------+-------------+----------+')

        total = {label: total + verified[label]  for label, total in self.verified_total.items()}
        if sum(total.values()) == 0:
            return

        for label, count in sorted(total.items(), key=lambda item: (-item[1], item[0])):
            if count:
                mean = f", mean: {self.verified_total[label] / ((self.runs - 1) * self.label2count[label]):.1f}" if self.runs > 1 else ""
                print(f"{label}: {verified[label]} schemes (total: {count}{mean})")

    def __get_verified_count(self) -> Dict[str, int]:
        verified = {label: 0 for label in self.verified_total}

        for solver in self.solvers:
            verified[solver.strategy.label] += solver.verified_count()

        return verified

    def __get_labels_count(self) -> Dict[str, int]:
        label2count = {label: 0 for label in self.verified_total}
        for strategy in self.strategies:
            label2count[strategy.label] += 1

        return label2count
