import argparse
import os.path

from src.decomposition_solver import DecompositionSolver
from src.decompositions import CyclicDecomposition, Decomposition
from src.entities.train_strategy import TrainStrategy
from src.utils import get_dtype, get_matmul_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="Dimension n", type=int, default=3)
    parser.add_argument("-m", help="Dimension m", type=int, default=3)
    parser.add_argument("-p", help="Dimension p", type=int, default=3)
    parser.add_argument("--rank", help="Decomposition rank", type=int, default=23)
    parser.add_argument("-s", help="Number of symmetric components (only when n=m=p)", type=int, default=0)
    parser.add_argument("-t", help="Number of cyclic triplets (only when n=m=p)", type=int, default=0)
    parser.add_argument("--data-type", help="Coefficients data type", choices=["complex64", "complex128", "float32", "float64"], default="float64")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=2048)
    parser.add_argument("--device", help="Torch device", type=str, default="cuda")
    parser.add_argument("--learning-rate", help="Learning rate", type=float, default=0.1)
    parser.add_argument("--steps", help="Number of steps per epoch", type=int, default=2000)
    parser.add_argument("--epochs", help="Number of epochs per experiment", type=int, default=4)
    parser.add_argument("--restarts", help="Number of restarts", type=int, default=4000)
    parser.add_argument("--log-period", help="Logging period (in steps)", type=int, default=100)
    parser.add_argument("-o", "--output-dir", help="Directory to save discovered decompositions", type=str, default="discovered_decompositions")
    args = parser.parse_args()

    use_cyclic = args.s > 0 or args.t > 0
    if use_cyclic:
        if not (args.n == args.m == args.p):
            parser.error("Cyclic symmetry parameters -s and -t require n=m=p (cubic tensors only)")

        rank = args.s + 3 * args.t
        if args.rank != rank:
            parser.error(f"Rank mismatch: --rank={args.rank}, but s + 3*t = {args.s} + 3*{args.t} = {rank}. With cyclic symmetry, rank must equal s + 3*t.")

    output_dir = os.path.join(args.output_dir, f"{args.n}x{args.m}x{args.p}", f"rank{args.rank}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting fast matmul decomposition search for {args.n}x{args.m}x{args.p} with rank {args.rank} using the default strategy")
    if use_cyclic:
        print(f"- decomposition type: cyclic symmetric (s = {args.s}, t = {args.t})")
    else:
        print(f"- decomposition type: usual")
    print(f"- data type: {args.data_type}")
    print(f"- batch size: {args.batch_size}")
    print(f"- device: {args.device}")
    print(f"- learning rate: {args.learning_rate}")
    print(f"- epochs: {args.epochs}")
    print(f"- steps per epoch: {args.steps}")
    print(f"- restarts: {args.restarts}")
    print(f"- logging period: {args.log_period} steps")
    print(f"- output directory: {output_dir}")

    strategy = TrainStrategy.default(learning_rate=args.learning_rate)

    dimension = f"({args.n}, {args.m}, {args.p}: {args.rank})"
    dtype = get_dtype(args.data_type)
    target_tensor = get_matmul_tensor(n=args.n, m=args.m, p=args.p, device=args.device, dtype=dtype)

    for restart in range(args.restarts):
        if use_cyclic:
            decomposition = CyclicDecomposition(n=args.n, s=args.s, t=args.t, rank=args.rank, dtype=dtype, batch_size=args.batch_size, device=args.device)
        else:
            decomposition = Decomposition(n=args.n, m=args.m, p=args.p, rank=args.rank, dtype=dtype, batch_size=args.batch_size, device=args.device)

        decomposition.initialize(scale=0.25)
        solver = DecompositionSolver(decomposition=decomposition, strategy=strategy, T=target_tensor, output_dir=output_dir)

        for epoch in range(args.epochs):
            for step in range(args.steps):
                loss = solver.step(step, args.steps, print_verified=False)

                if step % args.log_period == 0:
                    print(f"\n{dimension}: run {restart + 1}, epoch {epoch + 1} / {args.epochs}, step {step} / {args.steps}, loss: {loss}")
                    print("| reconstruction | rounded recons. (mean / min / best) | rationalization | magnitude |   balance   | verified |")
                    print("+----------------+-------------------------------------+-----------------+-----------+-------------+----------+")
                    solver.status()


if __name__ == '__main__':
    main()
