from typing import List, Tuple

from src.entities.train_parameters import TrainParameters


class TrainStrategy:
    def __init__(self, label: str, scales: List[int], learning_rate: float, optimizer_name: str = "adam"):
        self.label = label
        self.scales = scales
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.parameters = []
        self.default_parameters = TrainParameters.default()

    def add(self, parameters: TrainParameters):
        self.parameters.append(parameters)

    def get(self, step: int, steps: int) -> Tuple[TrainParameters, float]:
        start_step = 0

        for parameters in self.parameters:
            end_step = int(parameters.end_part * steps)
            if step < end_step:
                t = (step - start_step) / (end_step - start_step)
                return parameters, t

            start_step = end_step

        return self.default_parameters, step / steps
