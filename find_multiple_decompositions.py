import argparse
import math
import os.path
from typing import List

from src.decompositions import Decomposition
from src.entities.train_parameters import TrainParameters
from src.entities.train_strategy import TrainStrategy
from src.strategy_comparator import StrategyComparator
from src.utils import get_dtype, get_matmul_tensor


def init_strategies(learning_rate: float, max_abs_value: float = 1.5) -> List[TrainStrategy]:
    strategies = []

    rationalization_weights = [
        ("0.25t^2cos(5t^2)", "ternary", lambda t: 0.25 * t * t * ((1 + math.cos(2 * math.pi * t * t * 5)) / 2)),
    ]

    project_weights = [
        ("0", lambda t: 0.0),
        ("0.05t", lambda t: 0.05 * t),
    ]

    als_probabilities = [0.5, 0.75]
    end_parts = [0.4]
    total_steps = [2000]

    for rat_name, rat_type, rationalization_weight in rationalization_weights:
        for als_probability in als_probabilities:
            for end_part in end_parts:
                for project_name, project_weight in project_weights:
                    for steps in total_steps:
                        balance = TrainParameters(
                            end_part=end_part,
                            w_magnitude=lambda t: 0.01,
                            w_balance=lambda t: 0.01,
                            max_abs_value=max_abs_value,
                            als_probability=als_probability,
                            project_alpha=project_weight
                        )

                        rationalization = TrainParameters(
                            end_part=1.0,
                            w_rationalization=rationalization_weight,
                            w_magnitude=lambda t: 0.1 * t,
                            w_balance=lambda t: 0.01,
                            rationalization_type=rat_type,
                            max_abs_value=max_abs_value,
                            als_probability=als_probability,
                            project_alpha=project_weight
                        )

                        label = f"[ALS{als_probability:.2f}]-[E{end_part}]-[|{max_abs_value}|]-[P{project_name}]-[R{rat_name}-{rat_type}]-[lr{learning_rate}]-[ST{steps}]"
                        strategy = TrainStrategy(label=label, scales=[1, 2], learning_rate=learning_rate, steps=steps)
                        strategy.add(balance)
                        strategy.add(rationalization)
                        strategies.append(strategy)

    return strategies


def main():
    dimension2ranks = {
        (2, 4, 4): [26],
        (2, 4, 5): [32, 33],
        (2, 4, 6): [39],
        (3, 3, 5): [36],
        (3, 3, 6): [40, 42, 43, 44],
        (3, 4, 4): [38],
        (3, 4, 5): [47, 48],
        (3, 5, 5): [58],
        (4, 4, 4): [48, 49, 50]
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-type", help="Coefficients data type", choices=["complex64", "complex128", "float32", "float64"], default="float32")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=2048)
    parser.add_argument("--device", help="Torch device", type=str, default="cuda")
    parser.add_argument("--learning-rate", help="Learning rate", type=float, default=0.1)
    parser.add_argument("--epochs", help="Number of epochs per experiment", type=int, default=1)
    parser.add_argument("--log-period", help="Logging period (in steps)", type=int, default=100)
    parser.add_argument("-o", "--output-dir", help="Directory to save discovered decompositions", type=str, default="discovered_decompositions")
    args = parser.parse_args()

    print(f"Starting fast matmul decompositions")
    print(f"- data type: {args.data_type}")
    print(f"- batch size: {args.batch_size}")
    print(f"- device: {args.device}")
    print(f"- learning rate: {args.learning_rate}")
    print(f"- epochs: {args.epochs}")
    print(f"- logging period: {args.log_period} steps")
    print(f"- output directory: {args.output_dir}")

    dtype = get_dtype(args.data_type)
    strategies = init_strategies(args.learning_rate)
    comparators = []

    for (n, m, p), ranks in dimension2ranks.items():
        for rank in ranks:
            decomposition = Decomposition(n=n, m=m, p=p, rank=rank, dtype=dtype, batch_size=args.batch_size, device=args.device)
            target_tensor = get_matmul_tensor(n=n, m=m, p=p, device=args.device, dtype=dtype)
            output_dir = os.path.join(args.output_dir, f"{n}x{m}x{p}", f"rank{rank}")
            os.makedirs(output_dir, exist_ok=True)

            comparator = StrategyComparator(decomposition=decomposition, strategies=strategies, T=target_tensor, output_dir=output_dir)
            comparators.append(comparator)

    while True:
        for comparator in comparators:
            comparator.print_statistics()

        for comparator in comparators:
            comparator.run(epochs=args.epochs, log_period=args.log_period, print_verified=False)


if __name__ == '__main__':
    main()
