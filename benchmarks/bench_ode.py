"""基准 2：ODE 右端 + 解析 Jacobian（逐点求值场景）。

体系：合成刚性链式反应网络（组分数可调）。
对比 sympy.lambdify（Matrix+cse，对其最有利写法）在
单步 RHS / 单步 Jacobian / BDF 端到端积分三个层面的表现。

正确性：两条路线的 RHS/Jac 逐点对拍（rtol 1e-9），积分终值互检。

用法：python benchmarks/bench_ode.py [n_species] [--quick]
"""
from __future__ import annotations

import sys

import numpy as np
import sympy as sp

from _common import Table, ms, timeit_best, us


def make_system(n: int):
    """链式衰变 + 二阶耦合，速率跨 8 个量级（刚性）。"""
    ys = sp.symbols(f"y:{n}")
    k = [10.0 ** (4 - 8 * i / n) for i in range(n)]
    rhs = []
    for i in range(n):
        e = sp.Integer(0)
        if i > 0:
            e += k[i - 1] * ys[i - 1]
        e -= k[i] * ys[i]
        if i + 2 < n:
            e += 50.0 * ys[i + 2] * ys[0] - 50.0 * ys[i] * ys[0]
        rhs.append(e)
    return rhs, list(ys), [1.0] + [0.0] * (n - 1)


def main() -> int:
    import frontier as fr
    from scipy.integrate import solve_ivp

    quick = "--quick" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    n = int(args[0]) if args else (10 if quick else 40)

    rhs_exprs, state, y0 = make_system(n)

    # 路线 A：lambdify（Matrix + cse）
    f_lam = sp.lambdify(state, rhs_exprs, modules="numpy", cse=True)
    j_lam = sp.lambdify(state, sp.Matrix(rhs_exprs).jacobian(state),
                        modules="numpy", cse=True)
    rhs1 = lambda t, y: f_lam(*y)                       # noqa: E731
    jac1 = lambda t, y: np.asarray(j_lam(*y), float)    # noqa: E731

    # 路线 B：frontier
    ode = fr.compile_ode(rhs_exprs, state=state)

    # ---- 正确性 ----
    yt = np.asarray(y0) + 1e-4
    assert np.allclose(rhs1(0, yt), ode.rhs(0, yt), rtol=1e-9)
    assert np.allclose(jac1(0, yt), ode.jac(0, yt), rtol=1e-9)

    t_r1, _ = timeit_best(lambda: rhs1(0, yt))
    t_r2, _ = timeit_best(lambda: ode.rhs(0, yt))
    t_j1, _ = timeit_best(lambda: jac1(0, yt))
    t_j2, _ = timeit_best(lambda: ode.jac(0, yt))

    tbl = Table(f"ODE 逐点求值 @ {n} 组分刚性网络",
                ["lambdify", "frontier", "加速比"])
    tbl.row("单步 RHS", {"lambdify": us(t_r1), "frontier": us(t_r2),
                         "加速比": f"{t_r1 / t_r2:.1f}x"})
    tbl.row(f"单步 Jacobian ({n}x{n})",
            {"lambdify": us(t_j1), "frontier": us(t_j2),
             "加速比": f"{t_j1 / t_j2:.1f}x"})

    if not quick:
        span = (0, 100.0)
        t1, s1 = timeit_best(lambda: solve_ivp(
            rhs1, span, y0, jac=jac1, method="BDF", rtol=1e-8, atol=1e-10),
            repeats=3)
        t2, s2 = timeit_best(lambda: solve_ivp(
            ode.rhs, span, y0, jac=ode.jac, method="BDF", rtol=1e-8,
            atol=1e-10), repeats=3)
        assert s1.success and s2.success
        assert np.allclose(s1.y[:, -1], s2.y[:, -1], rtol=1e-5, atol=1e-12)
        tbl.row("BDF 端到端积分", {"lambdify": ms(t1), "frontier": ms(t2),
                                   "加速比": f"{t1 / t2:.1f}x"})

    tbl.print_table()
    print("  正确性：RHS/Jacobian 逐点对拍与积分终值互检均通过（rtol 1e-9 / 1e-5）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
