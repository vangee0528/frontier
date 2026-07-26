"""基准 4：规模扫描——加速比随表达式规模的变化规律。

固定场景（ODE 单步 RHS+Jacobian），扫组分数 n，展示两条曲线：
lambdify 成本随表达式数量线性增长，frontier 内核几乎不变。
这是"什么时候值得用 Frontier"的量化答案。

用法：python benchmarks/bench_scaling.py [--quick]
"""
from __future__ import annotations

import sys

import numpy as np
import sympy as sp

from _common import Table, timeit_best, us
from bench_ode import make_system


def main() -> int:
    import frontier as fr

    quick = "--quick" in sys.argv
    sizes = [5, 10, 20] if quick else [5, 10, 20, 40, 60]

    tbl = Table("规模扫描：单步 RHS + Jacobian 合计耗时",
                ["lambdify", "frontier", "加速比"])
    for n in sizes:
        rhs_exprs, state, y0 = make_system(n)
        f_lam = sp.lambdify(state, rhs_exprs, modules="numpy", cse=True)
        j_lam = sp.lambdify(state, sp.Matrix(rhs_exprs).jacobian(state),
                            modules="numpy", cse=True)
        ode = fr.compile_ode(rhs_exprs, state=state)

        yt = np.asarray(y0) + 1e-4
        assert np.allclose(f_lam(*yt), ode.rhs(0, yt), rtol=1e-9)

        t1, _ = timeit_best(lambda: (f_lam(*yt),
                                     np.asarray(j_lam(*yt), float)))
        t2, _ = timeit_best(lambda: (ode.rhs(0, yt), ode.jac(0, yt)))
        tbl.row(f"n = {n:>2d}（{n + n * n} 个表达式）",
                {"lambdify": us(t1), "frontier": us(t2),
                 "加速比": f"{t1 / t2:.1f}x"})

    tbl.print_table()
    print("  规律：lambdify 成本 ∝ 表达式数量；frontier 内核成本近乎常数")
    return 0


if __name__ == "__main__":
    sys.exit(main())
