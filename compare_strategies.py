import argparse
import math
import os.path
from typing import List

from src.entities.decomposition import Decomposition
from src.entities.train_parameters import TrainParameters
from src.entities.train_strategy import TrainStrategy
from src.strategy_comparator import StrategyComparator
from src.utils import get_dtype, get_matmul_tensor


def init_strategies(learning_rate: float, repeats: int = 2) -> List[TrainStrategy]:
    strategies = [
        TrainStrategy.default(learning_rate=learning_rate)
    ]

    rationalization_weights = [
        ("0.01", lambda t: 0.01),
        ("0.5t^2", lambda t: 0.5 * t * t),
        ("0.25t^2cos(5t^2)", lambda t: 0.25 * t * t * ((1 + math.cos(2 * math.pi * t * t * 5)) / 2)),
        ("0.50t^2cos(5t^2)", lambda t: 0.5 * t * t * ((1 + math.cos(2 * math.pi * t * t * 5)) / 2)),
    ]

    sparsity_weights = [
        ("0", lambda t: 0.0),
        ("0.008t", lambda t: 0.008 * t),
    ]

    project_weights = [
        ("0.010", lambda t: 0.010),
        ("0.05t", lambda t: 0.05 * t),
    ]

    als_probabilities = [0.5, 0.75]

    for rat_name, rationalization_weight in rationalization_weights:
        for als_probability in als_probabilities:
            for project_name, project_weight in project_weights:
                for sparsity_name, sparsity_weight in sparsity_weights:
                    balance = TrainParameters(
                        end_part=0.4,
                        w_rationalization=lambda t: 0,
                        w_sparsity=lambda t: 0,
                        w_magnitude=lambda t: 0.1 * t,
                        w_balance=lambda t: 0.01,
                        max_abs_value=2.0,
                        als_probability=als_probability,
                        project_alpha=project_weight
                    )

                    rationalization = TrainParameters(
                        end_part=1.0,
                        w_rationalization=rationalization_weight,
                        w_sparsity=sparsity_weight,
                        w_magnitude=lambda t: 0.1 * t,
                        w_balance=lambda t: 0.001,
                        rationalization_type="ternary",
                        sparsity_type="sqrt",
                        max_abs_value=2.0,
                        als_probability=als_probability,
                        project_alpha=project_weight
                    )

                    label = f"[ALS{als_probability:.2f}]-[P{project_name}]-[S{sparsity_name}]-[R{rat_name}]"
                    strategy = TrainStrategy(label=label, scales=[1, 2], learning_rate=learning_rate)
                    strategy.add(balance)
                    strategy.add(rationalization)
                    strategies.append(strategy)

    return strategies * repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="Dimension n", type=int, default=3)
    parser.add_argument("-m", help="Dimension m", type=int, default=3)
    parser.add_argument("-p", help="Dimension p", type=int, default=3)
    parser.add_argument("--rank", help="Decomposition rank", type=int, default=23)
    parser.add_argument("--data-type", help="Coefficients data type", choices=["complex64", "complex128", "float32", "float64"], default="float32")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=2048)
    parser.add_argument("--device", help="Torch device", type=str, default="cuda")
    parser.add_argument("--learning-rate", help="Learning rate", type=float, default=0.1)
    parser.add_argument("--steps", help="Number of steps per epoch", type=int, default=2000)
    parser.add_argument("--epochs", help="Number of epochs per experiment", type=int, default=2)
    parser.add_argument("--log-period", help="Logging period (in steps)", type=int, default=100)
    parser.add_argument("-o", "--output-dir", help="Directory to save discovered decompositions", type=str, default="discovered_decompositions")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f"{args.n}x{args.m}x{args.p}", f"rank{args.rank}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting fast matmul decomposition search for {args.n}x{args.m}x{args.p} with rank {args.rank} using different strategies")
    print(f"- data type: {args.data_type}")
    print(f"- batch size: {args.batch_size}")
    print(f"- device: {args.device}")
    print(f"- learning rate: {args.learning_rate}")
    print(f"- epochs: {args.epochs}")
    print(f"- steps per epoch: {args.steps}")
    print(f"- logging period: {args.log_period} steps")
    print(f"- output directory: {output_dir}")

    dtype = get_dtype(args.data_type)
    decomposition = Decomposition(n=args.n, m=args.m, p=args.p, rank=args.rank, dtype=dtype, batch_size=args.batch_size, device=args.device)

    strategies = init_strategies(learning_rate=args.learning_rate)
    target_tensor = get_matmul_tensor(n=args.n, m=args.m, p=args.p, device=args.device, dtype=dtype)

    comparator = StrategyComparator(decomposition=decomposition, strategies=strategies, T=target_tensor, output_dir=output_dir)

    while True:
        comparator.run(epochs=args.epochs, steps=args.steps, log_period=args.log_period, print_verified=False)


if __name__ == '__main__':
    main()
