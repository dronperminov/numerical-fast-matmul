import argparse
import os.path

from src.decomposition_solver import DecompositionSolver
from src.entities.decomposition import Decomposition
from src.entities.train_strategy import TrainStrategy
from src.utils import get_matmul_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="n", type=int, default=3)
    parser.add_argument("-m", help="m", type=int, default=3)
    parser.add_argument("-p", help="p", type=int, default=3)
    parser.add_argument("--rank", help="decomposition rank", type=int, default=23)
    parser.add_argument("--data-type", help="coefficients type", choices=["complex64", "complex128", "float32", "float64"], default="float32")
    parser.add_argument("--batch-size", help="batch size", type=int, default=2048)
    parser.add_argument("--device", help="torch device", type=str, default="cuda")
    parser.add_argument("--learning-rate", help="learning rate", type=int, default=0.1)
    parser.add_argument("--steps", help="steps per epoch", type=int, default=12000)
    parser.add_argument("--epoches", help="epoches number per experiment", type=int, default=150)
    parser.add_argument("--log-period", help="check period", type=int, default=100)
    parser.add_argument("-o", "--output-dir", help="directory for save discovered decompositions", type=str, default="discovered_decompositions")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f"{args.n}x{args.m}x{args.p}/rank{args.rank}")
    os.makedirs(output_dir, exist_ok=True)

    decomposition = Decomposition(n=args.n, m=args.m, p=args.p, rank=args.rank, data_type=args.data_type, batch_size=args.batch_size, device=args.device)
    decomposition.initialize()

    target_tensor = get_matmul_tensor(n=args.n, m=args.m, p=args.p, device=args.device)

    strategy = TrainStrategy(label="", scales=[1, 2], learning_rate=args.learning_rate, optimizer_name="adam")
    solver = DecompositionSolver(decomposition=decomposition, strategy=strategy, T=target_tensor, output_dir=output_dir)

    for epoch in range(args.epoches):
        for step in range(args.steps):
            loss = solver.train_step(step, args.steps)
            solver.verify()

            if step % args.log_period == 0:
                print(f"({args.n}, {args.m}, {args.p}: {args.rank}): epoch {epoch + 1}, step {step}, loss: {loss}")
                solver.status()


if __name__ == '__main__':
    main()
