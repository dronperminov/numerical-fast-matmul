import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class TrainParameters:
    end_part: float
    w_rationalization: Callable[[float], float] = lambda t: 0.0
    w_sparsity: Callable[[float], float] = lambda t: 0.0
    w_magnitude: Callable[[float], float] = lambda t: 0.0
    w_balance: Callable[[float], float] = lambda t: 0.0
    rationalization_type: str = "ternary"
    sparsity_type: str = "sqrt"
    max_abs_value: float = 3.0
    als_probability: float = 0.0
    project_alpha: Callable[[float], float] = lambda t: 0.0
    target_noise_std: Callable[[float], float] = lambda t: 0.0

    @staticmethod
    def default() -> "TrainParameters":
        return TrainParameters(
            end_part=1.0,
            w_rationalization=lambda t: 0.25 * t * t * (1 + math.cos(10 * math.pi * t * t)) / 2,
            w_magnitude=lambda t: 0.1 * t,
            w_balance=lambda t: 0.01,
            rationalization_type="ternary",
            sparsity_type="sqrt",
            max_abs_value=1.5,
            als_probability=0.75
        )
