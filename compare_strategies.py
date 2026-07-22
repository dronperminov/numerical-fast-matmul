
import argparse
import math
import os.path
from typing import List

from src.entities.decomposition import Decomposition
from src.entities.train_parameters import TrainParameters
from src.entities.train_strategy import TrainStrategy
from src.strategy_comparator import StrategyComparator
from src.utils import get_matmul_tensor, get_dtype


def init_strategies(learning_rate: float) -> List[TrainStrategy]:
    strategies = [
        TrainStrategy(label="default", scales=[1, 2], learning_rate=learning_rate)
    ]

    rationalization_weights = [
        ("0.01", lambda t: 0.01),
        ("0.5t", lambda t: 0.5*t),
        ("0.5t^2", lambda t: 0.5 * t * t),
        ("0.5t^2sin5", lambda t: 0.5 * t * t * (1 + math.sin(2 * math.pi * t * 5)) / 2),
        ("0.5t^2cos5", lambda t: 0.5 * t * t * (1 + math.cos(2 * math.pi * t * 5)) / 2),
    ]

    for rat_name, rationalization_weight in rationalization_weights:
        for als in [0.0, 0.4, 0.75]:
            for end_part in [0.2, 0.4]:
                parameters = TrainParameters(
                    end_part=1.0,
                    w_rationalization=rationalization_weight,
                    w_sparsity=lambda t: 0.008 * t,
                    w_magnitude=lambda t: 0.1 * t,
                    w_balance=lambda t: 0.01,
                    rationalization_type="ternary",
                    sparsity_type="sqrt",
                    max_abs_value=2.0,
                    als_probability=als
                )

                balance_only_als = TrainParameters(
                    end_part=end_part,
                    w_rationalization=lambda t: 0,
                    w_sparsity=lambda t: 0,
                    w_magnitude=lambda t: 0,
                    w_balance=lambda t: 0.01,
                    als_probability=als
                )
                strategy = TrainStrategy(label=f"als{als}-end_part{end_part}-rat{rat_name}", scales=[1, 2], learning_rate=learning_rate)
                strategy.add(balance_only_als)
                strategy.add(parameters)
                strategies.append(strategy)

    return strategies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="n", type=int, default=2)
    parser.add_argument("-m", help="m", type=int, default=4)
    parser.add_argument("-p", help="p", type=int, default=5)
    parser.add_argument("--rank", help="decomposition rank", type=int, default=32)
    parser.add_argument("--data-type", help="coefficients type", choices=["complex64", "complex128", "float32", "float64"], default="float32")
    parser.add_argument("--batch-size", help="batch size", type=int, default=1024)
    parser.add_argument("--device", help="torch device", type=str, default="cuda")
    parser.add_argument("--learning-rate", help="learning rate", type=int, default=0.1)
    parser.add_argument("--steps", help="steps per epoch", type=int, default=2000)
    parser.add_argument("--epoches", help="epoches number per experiment", type=int, default=2)
    parser.add_argument("--log-period", help="check period", type=int, default=100)
    parser.add_argument("-o", "--output-dir", help="directory for save discovered decompositions", type=str, default="discovered_decompositions")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f"{args.n}x{args.m}x{args.p}/rank{args.rank}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Start find fast matmul decomposition of {args.n}x{args.m}x{args.p} with rank {args.rank}")
    print(f"- data type: {args.data_type}")
    print(f"- batch size: {args.batch_size}")
    print(f"- device: {args.device}")
    print(f"- learning rate: {args.learning_rate}")
    print(f"- epoches: {args.epoches}")
    print(f"- steps per epoch: {args.steps}")
    print(f"- log period: {args.log_period}")
    print(f"- output directory: {output_dir}")
    print("")

    dtype = get_dtype(args.data_type)
    decomposition = Decomposition(n=args.n, m=args.m, p=args.p, rank=args.rank, dtype=dtype, batch_size=args.batch_size, device=args.device)
    decomposition.initialize()

    strategies = init_strategies(learning_rate=args.learning_rate)
    target_tensor = get_matmul_tensor(n=args.n, m=args.m, p=args.p, device=args.device)

    comparator = StrategyComparator(decomposition=decomposition, strategies=strategies, T=target_tensor, output_dir=output_dir)

    while True:
        comparator.reset()
        comparator.run(epoches=args.epoches, steps=args.steps, log_period=args.log_period, print_verified=True)


if __name__ == '__main__':
    main()
