# numerical-fast-matmul
Numerical search for low-rank tensor decompositions of matrix multiplication.

This repository contains an experimental framework for searching fast matrix multiplication algorithms using numerical optimization. The main
goal is to investigate different optimization strategies (gradient descent, alternating least squares, projection methods, regularization, etc.)
and eventually obtain an efficient solver capable of discovering decompositions for matrix multiplication tensors of various dimensions.

The project is intended as a research platform rather than a finished library. It allows experimenting with different optimization schedules,
comparing training strategies, and automatically saving discovered decompositions.

---

Multiplication of an $n \times m$ matrix by an $m \times p$ matrix can be represented as a third-order tensor

$$
T_{n,m,p}.
$$

A fast matrix multiplication algorithm corresponds to a low-rank decomposition

$$
T = \sum_{r=1}^{R} u_r \otimes v_r \otimes w_r.
$$

Finding such decompositions is a difficult non-convex optimization problem.

This repository explores numerical methods for solving this problem using:

* gradient-based optimization (Adam), 
* Alternating Least Squares (ALS), 
* coefficient projection, 
* rationalization losses, 
* sparsity regularization, 
* balance regularization, 
* randomized restarts, 
* and various combinations of these techniques.

## Installation

```bash
git clone https://github.com/dronperminov/numerical-fast-matmul
cd numerical-fast-matmul

pip install -r requirements.txt
```


## Scripts

The repository currently provides two main entry points.

### Search using the default strategy

```bash
python find_decomposition.py
```

This script repeatedly initializes random decompositions and optimizes them using the default training strategy.

Whenever a valid decomposition is found, it is automatically saved as JSON.

Usage:

```bash
python find_decomposition.py -n 3 -m 3 -p 3 --rank 23
```

### Compare optimization strategies

```bash
python compare_strategies.py -n 2 -m 4 -p 5 --rank 32
```

This script is intended for research.

Instead of repeatedly running a single optimization strategy, it cycles through a collection of different strategies and compares their
performance under identical conditions.

This makes it easy to answer questions such as:

- Does ALS improve convergence?
- Which rationalization schedule works best?
- Is projection useful?
- How important is sparsity regularization?
- Which combination discovers the largest number of valid decompositions?

## Command-line arguments

Both scripts share almost the same command-line interface.

| Argument             | Description                                                              | Default                     |
|----------------------|--------------------------------------------------------------------------|-----------------------------|
| `-n`                 | left matrix rows                                                         | `3`                         |
| `-m`                 | inner dimension                                                          | `3`                         |
| `-p`                 | right matrix columns                                                     | `3`                         |
| `--rank`             | target decomposition rank                                                | `23`                        |
| `--data-type`        | coefficients data type (`float32`, `float64`, `complex64`, `complex128`) | `float32`                   |
| `--batch-size`       | number of decompositions optimized simultaneously                        | `2048`                      |
| `--device`           | torch device                                                             | `cuda`                      |
| `--learning-rate`    | learning rate                                                            | `0.1`                       |
| `--steps`            | gradient steps per epoch                                                 | `2000`                      |
| `--epochs`           | number of epochs                                                         | varies                      |
| `--log-period`       | logging frequency                                                        | `100`                       |
| `-o`, `--output-dir` | directory for discovered decompositions                                  | `discovered_decompositions` |

Additional argument of `find_decomposition.py`:

| Argument     | Description                           |
|--------------|---------------------------------------|
| `--restarts` | Number of independent random restarts |


## Saved decompositions

Every verified decomposition is automatically written to JSON.

A decomposition is represented by three factor matrices:

```json
{
    "dimension": [3, 3, 3],
    "rank": 23, 
    "complexity": 90,
    "ring": "ZT",
    "strategy": "default",
    "u": [
        ...
    ],
    "v": [
        ...
    ],
    "w": [
        ...
    ]
}
```

where
* `u` contains coefficients for the `U` matrix,
* `v` contains coefficients for the `V` matrix,
* `w` contains coefficients for the `W` matrix.

Each row corresponds to one multiplication in the fast matrix multiplication algorithm.
