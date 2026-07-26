"""Frontier —— 符号-数值混合计算库。

用贴近 SymPy 的语法书写符号公式，C++ 内核完成求导与化简，
JIT 编译为机器码级批量数值函数。

    >>> import frontier as fr
    >>> x, y = fr.symbols("x y")
    >>> f = fr.sin(x * y) + x**2
    >>> fr.diff(f, x)
    y*cos(x*y) + 2*x
"""

from __future__ import annotations

from frontier._core import (  # noqa: F401  (re-export)
    CompileError,
    DomainError,
    Expr,
    FrontierError,
    as_expr,
    rational,
    symbol,
)
from frontier import _core
from frontier._compile import CompiledFunction, compile  # noqa: A004
from frontier._adapters import compile_fit, compile_objective, compile_ode
from frontier._interop import from_sympy, lambdify

__version__ = "1.0.0"

import math as _math

#: 圆周率 / 自然常数（f64 精度；符号层视作普通 Real 常量）
pi = as_expr(_math.pi)
E = as_expr(_math.e)

__all__ = [
    "Expr",
    "FrontierError",
    "DomainError",
    "CompileError",
    "symbol",
    "symbols",
    "as_expr",
    "rational",
    "diff",
    "grad",
    "jacobian",
    "hessian",
    "subs",
    "where",
    "pi",
    "E",
    "compile",
    "CompiledFunction",
    "from_sympy",
    "lambdify",
    "compile_ode",
    "compile_objective",
    "compile_fit",
    # 数学函数由注册表动态填充（见文件末尾），如 sin/cos/exp/...
]


def symbols(names: str) -> tuple[Expr, ...] | Expr:
    """按空格/逗号分隔创建一组符号：``x, y = fr.symbols("x y")``。

    支持 SymPy 的 range 语法：``fr.symbols("q:3")`` → (q0, q1, q2)，
    ``fr.symbols("a:d")`` → (a, b, c)。单个名字直接返回该符号。
    """
    parts = [p for p in names.replace(",", " ").split() if p]
    if not parts:
        raise ValueError("symbols(): no symbol names given")

    expanded: list[str] = []
    for p in parts:
        if ":" in p:
            base, _, spec = p.partition(":")
            if not spec:
                raise ValueError(f"symbols(): bad range syntax {p!r}")
            if spec.isdigit():  # q:3 → q0 q1 q2
                expanded.extend(f"{base}{i}" for i in range(int(spec)))
            elif len(spec) == 1 and spec.isalpha() and base and base[-1].isalpha():
                # 字母区间与 sympy 一致为闭区间：a:d → a b c d
                start, end = base[-1], spec
                prefix = base[:-1]
                if ord(end) < ord(start):
                    raise ValueError(f"symbols(): empty range {p!r}")
                expanded.extend(prefix + chr(c)
                                for c in range(ord(start), ord(end) + 1))
            else:
                raise ValueError(
                    f"symbols(): unsupported range syntax {p!r} "
                    "(supported: 'q:3' and 'a:z')")
        else:
            expanded.append(p)

    result = tuple(symbol(p) for p in expanded)
    return result[0] if len(result) == 1 else result


def diff(expr, var: Expr, *more_vars: Expr) -> Expr:
    """对 var 求偏导；传入多个变量时依序连续求导。"""
    result = _core.diff(as_expr(expr), var)
    for v in more_vars:
        result = _core.diff(result, v)
    return result


def grad(expr, variables) -> list[Expr]:
    """梯度：对每个变量求偏导，返回列表。"""
    return [_core.diff(as_expr(expr), v) for v in variables]


def jacobian(exprs, variables) -> list[list[Expr]]:
    """Jacobian 矩阵（列表的列表）：J[i][j] = ∂exprs[i]/∂variables[j]。"""
    return [[_core.diff(as_expr(e), v) for v in variables] for e in exprs]


def hessian(expr, variables) -> list[list[Expr]]:
    """Hessian 矩阵：H[i][j] = ∂²expr/∂vᵢ∂vⱼ（利用对称性只算上三角）。"""
    e = as_expr(expr)
    n = len(variables)
    grads = [_core.diff(e, v) for v in variables]
    H: list[list[Expr]] = [[None] * n for _ in range(n)]  # type: ignore[list-item]
    for i in range(n):
        for j in range(i, n):
            H[i][j] = H[j][i] = _core.diff(grads[i], variables[j])
    return H


def subs(expr, mapping: dict) -> Expr:
    """子表达式替换（也可用 expr.subs({...}) 方法形式）。"""
    return as_expr(expr).subs(mapping)


def where(cond, a, b) -> Expr:
    """条件选择：cond ≠ 0 时取 a，否则取 b（编译为无分支 select）。

    cond 通常来自比较运算：``fr.where(x > 0, x, -x)``。
    导数按分支传播（∂cond 视为 0，即忽略跳变点的 δ 函数）。
    """
    return _core.make_func("where", [as_expr(cond), as_expr(a), as_expr(b)])


def _install_registry_functions() -> None:
    """按 C++ 注册表自动生成模块级函数包装（sin、cos、exp……）。

    新函数在 C++ 侧注册后，Python 端无需任何改动即自动获得。
    """
    def make_wrapper(fname: str):
        def wrapper(*args):
            return _core.make_func(fname, [as_expr(a) for a in args])

        wrapper.__name__ = fname
        wrapper.__qualname__ = fname
        wrapper.__doc__ = f"符号函数 {fname}(...)（由 C++ FuncRegistry 提供）"
        return wrapper

    for fname in _core.function_names():
        # abs 等名字会遮蔽内置——只在 frontier 命名空间内，属预期行为
        globals()[fname] = make_wrapper(fname)
        if fname not in __all__:
            __all__.append(fname)


_install_registry_functions()
del _install_registry_functions
