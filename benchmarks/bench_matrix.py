"""基准 5：全对照矩阵——同一梯度负载横跨主流执行方案。

负载：f = sin(xy) + exp(-z²)(x+y)² + 100(y-x²)²，输出 [f, ∂f/∂x, ∂f/∂y, ∂f/∂z]。

对照（可用性自动探测，缺席的列注明跳过原因）：
  sympy-lambdify        默认
  sympy-lambdify-cse    cse=True
  numpy-hand            人肉求导 + 手写向量化
  numexpr               融合表达式（4 条逐条求值）
  numba-hand            人肉求导 + @njit 手写融合循环（单线程）
  jax-cpu               jit + 自动微分（sum 技巧取逐点梯度）
  frontier              默认 / fastmath / vecmath=False 三种配置

指标：
  cold      构造可调用对象耗时（含 frontier/lambdify 的编译）
  first     首次调用（含 numba/jax 的迟编译与追踪）
  hot-med   热执行中位数（20 次）
  hot-p95   热执行 95 分位
  amortize  编译成本摊销点：几次调用后追平 lambdify 默认路线
  max-rel   对 numpy-hand 参照的最大相对误差

输出分配口径：所有实现每次调用都分配新的输出数组（含 frontier；
它的 out= 复用接口另测，此表为默认口径的公平对比）。

用法：python benchmarks/bench_matrix.py [n_points] [--quick]
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np

from _common import Table


def measure(make, n_hot=20):
    """make() -> (callable, first_call_result)。返回全套计时。"""
    t0 = time.perf_counter()
    fn = make()
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = fn()
    first = time.perf_counter() - t0

    times = []
    for _ in range(n_hot):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    med = statistics.median(times)
    p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
    return dict(cold=cold, first=first, med=med, p95=p95, ref=ref)


def main() -> int:
    quick = "--quick" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    n = int(args[0]) if args else (200_000 if quick else 1_000_000)
    n_hot = 10 if quick else 20

    rng = np.random.default_rng(0)
    xs, ys, zs = (rng.uniform(0.1, 1.5, n) for _ in range(3))

    # ---- 参照实现：numpy 手写（人肉求导） ----
    def np_kernel():
        c = np.cos(xs * ys)
        e = np.exp(-zs**2)
        s = xs + ys
        dq = ys - xs**2
        f = np.sin(xs * ys) + e * s**2 + 100 * dq**2
        gx = ys * c + 2 * e * s - 400 * xs * dq
        gy = xs * c + 2 * e * s + 200 * dq
        gz = -2 * zs * e * s**2
        return f, gx, gy, gz

    ref_out = np_kernel()

    def max_rel(out):
        m = 0.0
        for a, b in zip(out, ref_out):
            denom = np.maximum(np.abs(b), 1e-300)
            m = max(m, float(np.max(np.abs(np.asarray(a) - b) / denom)))
        return m

    rows = []          # (name, metrics, note)

    def add_row(name, make, note=""):
        try:
            m = measure(make, n_hot)
            m["err"] = max_rel(m.pop("ref"))
            rows.append((name, m, note))
        except Exception as exc:                     # noqa: BLE001
            rows.append((name, None, f"跳过：{type(exc).__name__}: {exc}"[:60]))

    # ---- sympy lambdify（默认 / cse） ----
    import sympy as sp
    sx, sy, sz = sp.symbols("x y z")
    sf = (sp.sin(sx * sy) + sp.exp(-sz**2) * (sx + sy) ** 2
          + 100 * (sy - sx**2) ** 2)
    souts = [sf] + [sp.diff(sf, v) for v in (sx, sy, sz)]

    def mk_lam(cse):
        def make():
            fn = sp.lambdify((sx, sy, sz), souts, modules="numpy", cse=cse)
            return lambda: fn(xs, ys, zs)
        return make

    add_row("sympy-lambdify", mk_lam(False))
    add_row("sympy-lambdify-cse", mk_lam(True))
    add_row("numpy-hand", lambda: (lambda: np_kernel()))

    # ---- numexpr ----
    def mk_numexpr():
        import numexpr as ne
        exprs = [
            "sin(x*y) + exp(-z**2)*(x+y)**2 + 100*(y - x**2)**2",
            "y*cos(x*y) + 2*exp(-z**2)*(x+y) - 400*x*(y - x**2)",
            "x*cos(x*y) + 2*exp(-z**2)*(x+y) + 200*(y - x**2)",
            "-2*z*exp(-z**2)*(x+y)**2",
        ]
        ld = {"x": xs, "y": ys, "z": zs}
        ne.set_num_threads(1)   # 与其他单线程实现同口径
        return lambda: tuple(ne.evaluate(e, local_dict=ld) for e in exprs)

    add_row("numexpr(1t)", mk_numexpr)

    # ---- numba 手写融合循环 ----
    def mk_numba():
        import numba

        @numba.njit(cache=False, fastmath=False)
        def kern(x, y, z, f, gx, gy, gz):
            for i in range(x.size):
                c = np.cos(x[i] * y[i])
                e = np.exp(-z[i] * z[i])
                s = x[i] + y[i]
                dq = y[i] - x[i] * x[i]
                f[i] = np.sin(x[i] * y[i]) + e * s * s + 100 * dq * dq
                gx[i] = y[i] * c + 2 * e * s - 400 * x[i] * dq
                gy[i] = x[i] * c + 2 * e * s + 200 * dq
                gz[i] = -2 * z[i] * e * s * s

        def call():
            f = np.empty(n); gx = np.empty(n)
            gy = np.empty(n); gz = np.empty(n)
            kern(xs, ys, zs, f, gx, gy, gz)
            return f, gx, gy, gz
        return call

    add_row("numba-hand", mk_numba)

    # ---- jax cpu（可选） ----
    def mk_jax():
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)

        def scalar_f(x, y, z):
            return (jnp.sin(x * y) + jnp.exp(-z**2) * (x + y) ** 2
                    + 100 * (y - x**2) ** 2)

        # sum 技巧：逐点函数的批量梯度
        def batched(x, y, z):
            f = scalar_f(x, y, z)
            g = jax.grad(lambda a, b, c: jnp.sum(scalar_f(a, b, c)),
                         argnums=(0, 1, 2))(x, y, z)
            return (f, *g)

        fn = jax.jit(batched)
        jx, jy, jz = jnp.asarray(xs), jnp.asarray(ys), jnp.asarray(zs)
        return lambda: tuple(np.asarray(o) for o in fn(jx, jy, jz))

    add_row("jax-cpu", mk_jax)

    # ---- frontier 三配置 ----
    import frontier as fr
    fx, fy, fz = fr.symbols("x y z")
    ff = (fr.sin(fx * fy) + fr.exp(-fz**2) * (fx + fy) ** 2
          + 100 * (fy - fx**2) ** 2)
    fouts = [ff] + fr.grad(ff, [fx, fy, fz])

    def mk_frontier(**kw):
        def make():
            g = fr.compile(fouts, args=(fx, fy, fz), **kw)
            return lambda: g(xs, ys, zs)
        return make

    add_row("frontier", mk_frontier())
    add_row("frontier-fastmath", mk_frontier(fastmath=True))
    add_row("frontier-libm", mk_frontier(vecmath=False),
            note="vecmath=False，严格 libm")

    # ---- 汇总 ----
    base = next((m for name, m, _ in rows if name == "sympy-lambdify" and m), None)
    tbl = Table(f"执行方案矩阵 @ N = {n:,}（4 输出：f + 梯度）",
                ["cold", "first", "hot-med", "hot-p95", "amortize", "max-rel"])
    for name, m, note in rows:
        if m is None:
            tbl.row(name, {"cold": note})
            continue
        if base and base["med"] > m["med"]:
            saved = base["med"] - m["med"]
            amort = f"{(m['cold'] + m['first']) / saved:.0f} 次"
        else:
            amort = "—"
        tbl.row(name, {
            "cold": f"{m['cold']*1e3:.0f} ms",
            "first": f"{m['first']*1e3:.0f} ms",
            "hot-med": f"{m['med']*1e3:.1f} ms",
            "hot-p95": f"{m['p95']*1e3:.1f} ms",
            "amortize": amort,
            "max-rel": f"{m['err']:.1e}",
        })
    tbl.print_table()
    print("  amortize = (cold+first)/每次调用相对 sympy-lambdify 的节省；"
          "max-rel 以 numpy-hand 为参照")
    print("  所有实现均含输出数组分配（frontier 的 out= 复用接口另见文档）")

    # 正确性门槛：所有成功实现 max-rel < 1e-6（跨实现浮点路径差异容忍带）
    bad = [nm for nm, m, _ in rows if m and m["err"] > 1e-6]
    assert not bad, f"数值偏差超阈：{bad}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
