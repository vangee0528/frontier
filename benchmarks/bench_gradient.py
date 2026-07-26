"""旗舰基准：梯度流水线百万点批量求值。

场景：一个带超越函数的多元标量场，自动求梯度后在 N 个点上批量求值。
对比：
  1. frontier          —— fr.grad + fr.compile（本项目）
  2. frontier(fastmath)—— 同上，开 fast-math
  3. sympy.lambdify    —— SymPy 求导 + lambdify(modules='numpy')
  4. NumPy 手写        —— 人肉推导梯度后用 NumPy 向量化实现（参照上限）

用法：python benchmarks/bench_gradient.py [N]（默认 1_000_000）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import frontier as fr  # noqa: E402


def timeit_best(fn, *args, repeats: int = 7) -> float:
    """返回多次运行的最佳单次耗时（秒）。"""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    rng = np.random.default_rng(0)
    xs, ys, zs = (rng.uniform(0.1, 1.5, size=n) for _ in range(3))

    # ---- 目标标量场：f(x,y,z) = sin(xy) + exp(-z²)·(x+y)² + 100·(y-x²)² ----
    x, y, z = fr.symbols("x y z")
    f = fr.sin(x * y) + fr.exp(-(z**2)) * (x + y) ** 2 + 100 * (y - x**2) ** 2
    grad = fr.grad(f, [x, y, z])

    t0 = time.perf_counter()
    g_fr = fr.compile(grad, args=(x, y, z))
    t_compile = time.perf_counter() - t0
    g_fm = fr.compile(grad, args=(x, y, z), fastmath=True)

    # ---- SymPy 对照 ----
    import sympy as sp

    sx, sy, sz = sp.symbols("x y z")
    sf = sp.sin(sx * sy) + sp.exp(-(sz**2)) * (sx + sy) ** 2 + 100 * (sy - sx**2) ** 2
    sgrad = [sp.diff(sf, v) for v in (sx, sy, sz)]
    g_sp = sp.lambdify((sx, sy, sz), sgrad, modules="numpy")

    # ---- NumPy 手写梯度（人肉求导，向量化实现的参照上限）----
    def g_np(a, b, c):
        cos_ab = np.cos(a * b)
        e = np.exp(-(c**2))
        s = a + b
        dq = b - a**2
        gx = b * cos_ab + 2 * e * s - 400 * a * dq
        gy = a * cos_ab + 2 * e * s + 200 * dq
        gz = -2 * c * e * s**2
        return gx, gy, gz

    # ---- 正确性互检 ----
    r_fr = g_fr(xs, ys, zs)
    r_sp = g_sp(xs, ys, zs)
    r_np = g_np(xs, ys, zs)
    for a, b, c in zip(r_fr, r_sp, r_np):
        assert np.allclose(a, b, rtol=1e-9), "frontier vs sympy mismatch"
        assert np.allclose(a, c, rtol=1e-9), "frontier vs numpy mismatch"

    # ---- 计时 ----
    t_fr = timeit_best(g_fr, xs, ys, zs)
    t_fm = timeit_best(g_fm, xs, ys, zs)
    t_sp = timeit_best(g_sp, xs, ys, zs)
    t_np = timeit_best(g_np, xs, ys, zs)

    print(f"\n梯度批量求值 @ N = {n:,}（3 输入 → 3 输出，best of 7）\n")
    rows = [
        ("frontier (fr.compile)", t_fr),
        ("frontier (fastmath)", t_fm),
        ("sympy lambdify(numpy)", t_sp),
        ("NumPy 手写梯度", t_np),
    ]
    for name, t in rows:
        speedup = t_sp / t
        print(f"  {name:<24s} {t * 1e3:9.2f} ms   {speedup:6.2f}x vs lambdify")
    print(f"\n  frontier 编译耗时（符号求导+CSE+JIT）: {t_compile * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
