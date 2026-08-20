# numerical-fast-matmul
Numerical search for low-rank tensor decompositions of matrix multiplication.

This repository contains an experimental framework for searching fast matrix multiplication algorithms using numerical optimization. The main
goal is to investigate different optimization strategies (gradient descent, alternating least squares, projection methods, regularization, etc.)
and eventually obtain an efficient solver capable of discovering decompositions for matrix multiplication tensors of various dimensions.

The project is intended as a research platform rather than a finished library. It allows experimenting with different optimization schedules,
comparing training strategies, and automatically saving discovered decompositions.

Multiplication of an $n \times m$ matrix by an $m \times p$ matrix can be represented as a third-order tensor

$$
T_{nm,mp,pn}.
$$

A fast matrix multiplication algorithm corresponds to a low-rank decomposition

$$
T = \sum_{l=1}^{r} u_l \otimes v_l \otimes w_l.
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

### Search using the default strategy (`find_decomposition.py`)

This script repeatedly initializes random decompositions and optimizes them using the default training strategy.
Whenever a valid decomposition is found, it is automatically saved as JSON.

Usage:
```bash
python find_decomposition.py -n 3 -m 3 -p 3 --rank 23
```

### Compare optimization strategies (`compare_strategies.py`)

Instead of repeatedly running a single optimization strategy, it cycles through a collection of different strategies and compares their
performance under identical conditions.

This makes it easy to answer questions such as:

- Does ALS improve convergence?
- Which rationalization schedule works best?
- Is projection useful?
- How important is sparsity regularization?
- Which combination discovers the largest number of valid decompositions?

Usage:
```bash
python compare_strategies.py -n 2 -m 4 -p 5 --rank 32
```

### Command-line arguments

Both scripts share almost the same command-line interface.

* `-n` — left matrix rows (`3` by default);
* `-m` — inner dimension (`3` by default);
* `-p` — right matrix columns (`3` by default);
* `--rank` — target decomposition rank (`23` by default);
* `--data-type` — coefficients data type (`float32`, `float64`, `complex64` or `complex128`, `float32` by default);
* `--batch-size` — number of decompositions optimized simultaneously (`2048` by default);
* `--device` — torch device (`cuda` by default);
* `--learning-rate` — learning rate (`0.1` by default);
* `--steps` — gradient steps per epoch (`2000` by default);
* `--epochs` — number of epochs (s by default);
* `--log-period` — logging frequency (`100` by default);
* `-o` — directory for discovered decompositions (`discovered_decompositions` by default).

Additional argument of `find_decomposition.py`:
* `--restarts` — number of independent random restarts.


### Search for multiple dimensions and ranks (`find_multiple_decompositions.py`)

This script is designed for systematic search across multiple tensor dimensions and ranks simultaneously. 
It initializes a set of optimization strategies and applies them to multiple decomposition problems 
($n \times m \times p$ with different ranks) in parallel, cycling through them until stopped.

Key features:
- **Multi-target search**: searches for decompositions for multiple `(n, m, p)` dimensions and ranks.
- **Strategy battery**: uses a predefined set of strategies with different combinations of ALS, rationalization, and projection parameters.
- **Automatic directory organization**: saves discovered decompositions in structured subdirectories (`output_dir/{n}x{m}x{p}/rank{r}/`).
- **Continuous execution**: runs in an infinite loop, allowing it to keep searching until manually stopped, accumulating discoveries over time.

Usage:
```bash
python find_multiple_decompositions.py
```

The script has a hardcoded dictionary `dimension2ranks` that specifies which dimensions and ranks to search for. You can modify this dictionary
to add new target configurations:

```python
dimension2ranks = {
    (2, 4, 4): [26],
    (2, 4, 5): [32, 33],
    (2, 4, 6): [39],
    (3, 3, 5): [36, 37]
    # Add more configurations here
}
```

#### Command-line arguments

The script accepts the same arguments as `find_decomposition.py`, except for `--restarts`:
- `--data-type` — coefficients data type (`float32`, `float64`, `complex64` or `complex128`, `float32` by default);
- `--batch-size` — number of decompositions optimized simultaneously (`2048` by default);
- `--device` — torch device (`cuda` by default);
- `--learning-rate` — optimizer learning rate (`0.1` by default);
- `--epochs` — number of epochs per experiment (`1` by default);
- `--log-period` — logging frequency (`100` by default);
- `-o` — directory for discovered decompositions (`discovered_decompositions` by default).


## Saved decompositions

Every verified decomposition is automatically saved as a JSON file.

Example:

```json
{
    "dimension": [3, 3, 3],
    "rank": 23,
    "complexity": 90,
    "ring": "ZT",
    "strategy": "default",
    "u": [
        [0, 0, 0, -1, 0, 0, 0, 0, 0],
        [1, 0, 0,  0, 0, 0, 0, 0, 0],
        ...
    ],
    "v": [
        [0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0,  1, 0, 0, 0, 0, 0, 0],
        ...
    ],
    "w": [
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0],
        ...
    ]
}
```

The fields have the following meaning:
- `dimension` — dimensions `(n, m, p)` of the matrix multiplication tensor, corresponding to multiplication of
an $n \times m$ matrix by an $m \times p$ matrix.

- `rank` — rank of the decomposition, i.e. the number of scalar multiplications used by the algorithm.

- `complexity` — number of scalar additions/subtractions required by the decomposition before applying any common-subexpression elimination or
other algebraic optimizations.

- `ring` — the smallest coefficient ring containing all decomposition coefficients:
  - `ZT` — ternary integers `{-1, 0, 1}`;
  - `Z` — arbitrary integers (with at least one coefficient outside `{-1, 0, 1}`);
  - `Q` — rational coefficients;
  - `C` — complex coefficients.

- `strategy` — name of the optimization strategy that discovered the decomposition.

- `u` — matrix `U`. Each **row** defines one linear combination of entries of the left input matrix used to compute one scalar multiplication.

- `v` — matrix `V`. Each **row** defines one linear combination of entries of the right input matrix used to compute one scalar multiplication.

- -`w` — matrix `W`. Each **column** specifies how the scalar multiplications are combined to reconstruct one entry of the output matrix.
