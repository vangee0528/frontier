# Frontier

English | [简体中文](README.zh-CN.md)

**A compiled execution layer for symbolic math**: write formulas in
SymPy-like syntax, compile them into machine-code batch kernels.
A drop-in replacement for `sympy.lambdify` wherever you *derive once,
evaluate a million times* — fitting, optimization, simulation, ODEs.

```python
import frontier as fr
import numpy as np

x, y, z = fr.symbols("x y z")
f = fr.sin(x * y) + fr.exp(-z**2) * (x + y) ** 2

g = fr.compile(fr.grad(f, [x, y, z]), args=(x, y, z))   # symbolic grad -> machine code

xs, ys, zs = (np.random.rand(1_000_000) for _ in range(3))
gx, gy, gz = g(xs, ys, zs)
```

Migrating an existing SymPy project is one line:

```python
f = fr.lambdify(args, exprs)     # same signature as sympy.lambdify
```

## Features

- **Fast** — single-pass fused loops, free CSE, SIMD-vectorized
  transcendentals, and loop-invariant parameter hoisting. Typically
  3–36× over `sympy.lambdify` in batch workloads, and faster than
  hand-written NumPy in most of them (numbers and reproduction:
  [performance guide](docs/performance.md))
- **Automatic derivatives** — `diff` / `grad` / `jacobian` / `hessian`
  with analytic precision, no finite-difference noise;
  `compile_ode` / `compile_objective` / `compile_fit` plug straight
  into SciPy solvers
- **Correct at the symbolic layer** — exact integer/rational
  arithmetic, canonical-on-construction simplification, global
  expression interning (structural equality is O(1))
- **Engineering-ready** — thread-safe, picklable
  (multiprocessing/joblib), on-disk compile cache, actionable errors
- **Light dependencies** — `numpy` + `llvmlite`, both plain pip installs

## Installation

```bash
pip install frontier-symbolic
```

Building from source: see the [quickstart](docs/quickstart.md).
Python ≥ 3.10; Linux / macOS / Windows.

## Documentation

| | |
| --- | --- |
| [Quickstart](docs/quickstart.md) | install + five-minute tutorial |
| [User guide](docs/guide.md) | concepts, compile options, SymPy migration, SciPy integration |
| [API reference](docs/api.md) | every public interface |
| [Performance guide](docs/performance.md) | benchmark numbers, tuning, lambdify differences |
| [Case studies](docs/case-studies.md) | five real-ecosystem migration patterns with measurements |
| [Internals](docs/internals.md) | architecture, invariants, extension points |

Runnable examples live in [examples/](examples/). Documentation is
currently written in Chinese; an English translation is planned —
the code, API names, and error messages are English throughout.

## Scope (honest edition)

Frontier is not a full CAS: no integration, no equation solving, no
trig identity rewriting — do those in SymPy and feed the result to
`fr.from_sympy`. The numeric domain is f64 real. For small
expressions (≤5 variables) evaluated one point at a time, Frontier
ties with lambdify; the advantage grows with expression size and
batch size.

## Correctness

- Cross-validation against SymPy (values + gradients, rtol 1e-9)
  runs with the test suite, including a differential-fuzzing smoke
  batch (`tests/fuzz/`);
- Before release, 4000+ random expressions were three-way arbitrated
  (Frontier vs `sympy.lambdify` vs 30-digit SymPy evalf) with zero
  unresolved defects;
- 102 C++ assertions + 70 Python tests;
- Reproduce the benchmark suite in-repo:
  `python benchmarks/run_all.py --quick` (batch / pointwise /
  fitting / scaling, each with built-in correctness assertions).

## License

MIT — see [LICENSE](LICENSE). Third-party notices:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Release history: [CHANGELOG](CHANGELOG.md).
