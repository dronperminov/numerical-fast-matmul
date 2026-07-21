import argparse
import os.path

from src.decomposition_solver import DecompositionSolver
from src.entities.decomposition import Decomposition
from src.entities.train_parameters import TrainParameters
from src.entities.train_strategy import TrainStrategy
from src.utils import get_matmul_tensor, get_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="n", type=int, default=2)
    parser.add_argument("-m", help="m", type=int, default=4)
    parser.add_argument("-p", help="p", type=int, default=5)
    parser.add_argument("--rank", help="decomposition rank", type=int, default=32)
    parser.add_argument("--data-type", help="coefficients type", choices=["complex64", "complex128", "float32", "float64"], default="float32")
    parser.add_argument("--batch-size", help="batch size", type=int, default=2048)
    parser.add_argument("--device", help="torch device", type=str, default="cuda")
    parser.add_argument("--learning-rate", help="learning rate", type=int, default=0.1)
    parser.add_argument("--steps", help="steps per epoch", type=int, default=2000)
    parser.add_argument("--epoches", help="epoches number per experiment", type=int, default=4)
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

    strategy = TrainStrategy(label="", scales=[1, 2], learning_rate=args.learning_rate, optimizer_name="adam")
    strategy.add(TrainParameters.balance_only(end_part=0.4, weight=0.01))
    strategy.add(TrainParameters.default())

    dtype = get_dtype(args.data_type)

    decomposition = Decomposition(n=args.n, m=args.m, p=args.p, rank=args.rank, dtype=dtype, batch_size=args.batch_size, device=args.device)
    decomposition.initialize()

    target_tensor = get_matmul_tensor(n=args.n, m=args.m, p=args.p, device=args.device)

    solver = DecompositionSolver(decomposition=decomposition, strategy=strategy, T=target_tensor, output_dir=output_dir)

    for epoch in range(args.epoches):
        for step in range(args.steps):
            loss = solver.step(step, args.steps)

            if step % args.log_period == 0:
                print(f"({args.n}, {args.m}, {args.p}: {args.rank}): epoch {epoch + 1}, step {step} / {args.steps}, loss: {loss}")
                solver.status()


if __name__ == '__main__':
    main()
