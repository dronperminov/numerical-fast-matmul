import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class TrainParameters:
    end_part: float
    w_rationalization: Callable[[float], float]
    w_sparsity: Callable[[float], float]
    w_magnitude: Callable[[float], float]
    w_balance: Callable[[float], float]
    rationalization_type: str = "ternary"
    sparsity_type: str = "sqrt"
    max_abs_value: float = 3.0

    @staticmethod
    def default() -> TrainParameters:
        return TrainParameters(
            end_part=1.0,
            w_rationalization=lambda t: 0.5 * t * t * (1 + math.sin(10 * math.pi * t * t)) / 2,
            w_sparsity=lambda t: 0.008 * t,
            w_magnitude=lambda t: 0.1 * t,
            w_balance=lambda t: 0.01,
            rationalization_type="random",
            sparsity_type="sqrt",
            max_abs_value=2.0
        )
