"""基准 3：非线性最小二乘拟合（实用性验证的核心场景）。

场景：衰减振荡曲线 y = A·exp(-γt)·sin(ωt+φ) + c，5 参数。
四条路线在同一数据、同一初值上做 scipy least_squares：

  A. 有限差分 Jacobian（最常见的偷懒写法）
  B. 手推手写解析 Jacobian（正确性风险高的传统做法）
  C. sympy 求导 + lambdify
  D. frontier（fr.compile_fit，自动导数 + uniform 参数）

正确性：四条路线必须收敛到同一参数（互差 < 1e-6）。

用法：python benchmarks/bench_fitting.py [n_points] [--quick]
"""
from __future__ import annotations

import sys

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from _common import Table, ms, timeit_best


def main() -> int:
    import frontier as fr

    quick = "--quick" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    n = int(args[0]) if args else (100_000 if quick else 500_000)

    rng = np.random.default_rng(7)
    TRUE = np.array([2.0, 0.30, 5.0, 0.50, 0.10])   # A γ ω φ c
    P0 = np.array([1.0, 0.10, 4.0, 0.00, 0.00])
    t = np.linspace(0.0, 10.0, n)
    y = (TRUE[0] * np.exp(-TRUE[1] * t) * np.sin(TRUE[2] * t + TRUE[3])
         + TRUE[4] + rng.normal(0, 0.02, n))

    # ---- A. 有限差分 ----
    def model_np(p):
        return p[0] * np.exp(-p[1] * t) * np.sin(p[2] * t + p[3]) + p[4]

    res_np = lambda p: model_np(p) - y                          # noqa: E731

    # ---- B. 手写解析 Jacobian ----
    def jac_hand(p):
        e = np.exp(-p[1] * t)
        s = np.sin(p[2] * t + p[3])
        co = np.cos(p[2] * t + p[3])
        return np.stack([e * s, -p[0] * t * e * s, p[0] * t * e * co,
                         p[0] * e * co, np.ones_like(t)], axis=1)

    # ---- C. sympy + lambdify ----
    sA, sg, sw, sphi, sc, st = sp.symbols("A g w phi c t")
    smodel = sA * sp.exp(-sg * st) * sp.sin(sw * st + sphi) + sc
    spars = [sA, sg, sw, sphi, sc]
    f_sp = sp.lambdify((*spars, st), smodel, modules="numpy")
    j_sp = sp.lambdify((*spars, st), [sp.diff(smodel, v) for v in spars],
                       modules="numpy")
    res_sp = lambda p: f_sp(*p, t) - y                           # noqa: E731

    def jac_sp(p):
        cols = j_sp(*p, t)
        return np.stack([np.broadcast_to(np.asarray(cc, float), t.shape)
                         for cc in cols], axis=1)

    # ---- D. frontier ----
    fA, fg, fw, fphi, fc, ft = fr.symbols("A g w phi c t")
    fmodel = fA * fr.exp(-fg * ft) * fr.sin(fw * ft + fphi) + fc
    res_fr, jac_fr = fr.compile_fit(fmodel, [fA, fg, fw, fphi, fc], ft, t, y)

    routes = [
        ("A. 有限差分", res_np, "2-point"),
        ("B. 手写解析 Jac", res_np, jac_hand),
        ("C. sympy lambdify", res_sp, jac_sp),
        ("D. frontier", res_fr, jac_fr),
    ]

    tbl = Table(f"衰减振荡拟合 @ N = {n:,}，5 参数",
                ["拟合耗时", "nfev", "参数误差"])
    solutions = []
    for name, res, jac in routes:
        tt, fit = timeit_best(
            lambda r=res, j=jac: least_squares(r, P0, jac=j, method="trf"),
            repeats=2 if quick else 3)
        solutions.append(fit.x)
        tbl.row(name, {"拟合耗时": ms(tt), "nfev": str(fit.nfev),
                       "参数误差": f"{np.max(np.abs(fit.x - TRUE)):.1e}"})

    # ---- 正确性：四条路线同解 ----
    for x in solutions[1:]:
        assert np.allclose(x, solutions[0], atol=1e-6)

    tbl.print_table()
    print("  正确性：四条路线收敛到同一参数（互差 < 1e-6）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
