"""SciPy 生态一行接入适配器。

把符号模型打包成 scipy.integrate / scipy.optimize 需要的可调用组合，
自动求导 + JIT 编译，签名严格对齐 scipy 的调用约定：

- :func:`compile_ode`        → ``solve_ivp(rhs, ..., jac=jac)``
- :func:`compile_objective`  → ``minimize(f, jac=f.grad, hess=f.hess)``
- :func:`compile_fit`        → ``least_squares(res, jac=jac)``
"""

from __future__ import annotations

import ctypes

import numpy as np

from frontier import _core
from frontier._compile import CompiledFunction
from frontier._interop import from_sympy


def _to_fr(exprs):
    return [from_sympy(e) for e in exprs]


class OdeFunctions:
    """compile_ode 的返回值：rhs / jac 成对，直接喂 solve_ivp。

    注意：单个 OdeFunctions 实例内部复用预分配缓冲（积分器逐步调用
    是串行的），**实例不可跨线程共享**——并行积分请每线程各建一个
    （compile 结果有进程内缓存，重复构建近乎免费）。
    """

    def __init__(self, rhs_exprs, state, t, params, **compile_kws):
        state = [from_sympy(s) for s in state]
        rhs = _to_fr(rhs_exprs)
        n = len(state)
        if len(rhs) != n:
            raise ValueError(f"compile_ode: {len(rhs)} equations for {n} states")

        self._params = [from_sympy(p) for p in params]
        t_sym = from_sympy(t) if t is not None else _core.symbol("_t_unused")
        args = [t_sym, *state, *self._params]

        jac_exprs = [_core.diff(e, s) for e in rhs for s in state]
        self._rhs = CompiledFunction(rhs, args, **compile_kws)
        self._jac = CompiledFunction(jac_exprs, args, **compile_kws)
        self._n = n

        # 专用快路径：一块输入缓冲 + 两块输出缓冲 + 预构建指针数组，
        # 每步只做「填缓冲 → 一次 C 调用 → copy」
        n_in = 1 + n + len(self._params)
        self._in = np.empty(n_in)
        self._in_ptrs = (ctypes.c_void_p * n_in)(
            *(self._in.ctypes.data + 8 * i for i in range(n_in)))
        self._rhs_out = np.empty(n)
        self._rhs_ptrs = (ctypes.c_void_p * n)(
            *(self._rhs_out.ctypes.data + 8 * i for i in range(n)))
        self._jac_out = np.empty((n, n))
        self._jac_ptrs = (ctypes.c_void_p * (n * n))(
            *(self._jac_out.ctypes.data + 8 * i for i in range(n * n)))

    def _fill(self, t, y, p) -> None:
        if len(p) != len(self._params):
            raise TypeError(
                f"expected {len(self._params)} parameter value(s) "
                f"({[str(s) for s in self._params]}), got {len(p)} — "
                "pass them via solve_ivp(..., args=(...))")
        buf = self._in
        buf[0] = t
        buf[1:1 + self._n] = y
        if p:
            buf[1 + self._n:] = p

    def rhs(self, t, y, *p):
        self._fill(t, y, p)
        self._rhs._kernel.cfunc(self._in_ptrs, self._rhs_ptrs, 1)
        return self._rhs_out.copy()

    def jac(self, t, y, *p):
        self._fill(t, y, p)
        self._jac._kernel.cfunc(self._in_ptrs, self._jac_ptrs, 1)
        return self._jac_out.copy()

    def __iter__(self):  # (rhs, jac) 解包
        yield self.rhs
        yield self.jac


def compile_ode(rhs_exprs, state, t=None, params=(), **compile_kws) -> OdeFunctions:
    """符号 ODE 右端 → ``solve_ivp`` 可用的 (rhs, jac) 组合。

    示例（x, v, k 为 fr.symbols 创建的符号）::

        f = fr.compile_ode([v, -k*x], state=[x, v], params=[k])
        solve_ivp(f.rhs, (0, 10), y0, jac=f.jac, args=(2.0,), method="Radau")
    """
    return OdeFunctions(rhs_exprs, state, t, params, **compile_kws)


class Objective:
    """compile_objective 的返回值：f(x) / f.grad(x) / f.hess(x)。"""

    def __init__(self, expr, variables, params, **compile_kws):
        from frontier import hessian as fr_hessian

        variables = [from_sympy(v) for v in variables]
        e = from_sympy(expr)
        self._params = [from_sympy(p) for p in params]
        args = [*variables, *self._params]
        n = len(variables)

        grads = [_core.diff(e, v) for v in variables]
        hess = [h for row in fr_hessian(e, variables) for h in row]
        self._f = CompiledFunction([e], args, **compile_kws)
        self._g = CompiledFunction(grads, args, **compile_kws)
        self._h = CompiledFunction(hess, args, **compile_kws)
        self._n = n

    def __call__(self, x, *p) -> float:
        return float(self._f.eval_scalars(*x, *p)[0])

    def grad(self, x, *p) -> np.ndarray:
        return self._g.eval_scalars(*x, *p)

    def hess(self, x, *p) -> np.ndarray:
        return self._h.eval_scalars(*x, *p).reshape(self._n, self._n)


def compile_objective(expr, variables, params=(), **compile_kws) -> Objective:
    """符号目标函数 → ``minimize(f, x0, jac=f.grad, hess=f.hess)``。"""
    return Objective(expr, variables, params, **compile_kws)


class FitFunctions:
    """compile_fit 的返回值：residual(p) / jac(p)，闭包住数据数组。"""

    def __init__(self, model, params, x, x_data, y_data, **compile_kws):
        params = [from_sympy(p) for p in params]
        xs = from_sympy(x)
        m = from_sympy(model)
        args = [*params, xs]
        grads = [_core.diff(m, p) for p in params]

        kws = dict(compile_kws)
        kws.setdefault("uniform", params)  # 拟合参数天然 uniform
        self._val = CompiledFunction([m], args, **kws)
        self._jac = CompiledFunction(grads, args, **kws)
        self.x_data = np.ascontiguousarray(x_data, dtype=np.float64)
        self.y_data = np.ascontiguousarray(y_data, dtype=np.float64)

    def residual(self, p):
        return self._val.eval_stacked(*p, self.x_data)[0] - self.y_data

    def jac(self, p):
        # scipy least_squares 期望 (m, n)：行 = 数据点
        return self._jac.eval_stacked(*p, self.x_data).T

    def __iter__(self):  # (residual, jac) 解包
        yield self.residual
        yield self.jac


def compile_fit(model, params, x, x_data, y_data, **compile_kws) -> FitFunctions:
    """符号模型 + 数据 → ``least_squares(res, p0, jac=jac)``。

    示例（a, b, t 为符号，t_data/y_data 为数据数组）::

        res, jac = fr.compile_fit(a*fr.exp(-b*t), [a, b], t, t_data, y_data)
        fit = least_squares(res, p0, jac=jac)
    """
    return FitFunctions(model, params, x, x_data, y_data, **compile_kws)
