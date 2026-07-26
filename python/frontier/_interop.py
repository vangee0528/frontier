"""SymPy 互操作：迁移入口。

已有项目的符号推导代码（SymPy）一行不改，把最后一步的
``sympy.lambdify(args, exprs)`` 换成 ``fr.lambdify(args, exprs)`` 即可。

- ``from_sympy``：SymPy 表达式 → Frontier 表达式（迭代式遍历 + memo，
  不受 Python 递归深度限制，共享子树只转换一次）
- ``lambdify``：仿 ``sympy.lambdify`` 签名的编译门面
"""

from __future__ import annotations

from typing import Any

from frontier import _core

# SymPy 函数名 → Frontier 注册表名（大小写/别名差异）
_FUNC_NAME_MAP = {
    "Abs": "abs",
    "sign": "sign",
    "Max": "max",
    "Min": "min",
    "ceiling": "ceil",
}


def _registry_name(sympy_func_name: str) -> str | None:
    name = _FUNC_NAME_MAP.get(sympy_func_name, sympy_func_name)
    return name if name in _registry_names() else None


_names_cache: set[str] | None = None


def _registry_names() -> set[str]:
    global _names_cache
    if _names_cache is None:
        _names_cache = set(_core.function_names())
    return _names_cache


def _convert_node(e, cargs: list, sp) -> Any:
    """把一个 sympy 节点（子节点已转换为 cargs）转为 fr 表达式。"""
    if e.is_Symbol:
        return _core.symbol(e.name)
    if e.is_Integer:
        v = int(e)
        if -(2**63) <= v < 2**63:
            return _core.as_expr(v)
        return _core.as_expr(float(e))  # 超出 int64：降级浮点
    if e.is_Rational:  # Integer 已在上面截获；含 Half 等
        return _core.rational(int(e.p), int(e.q))
    if e.is_Float:
        return _core.as_expr(float(e))
    if isinstance(e, sp.NumberSymbol):  # pi、E、EulerGamma…
        return _core.as_expr(float(e))
    if e is sp.I or e is sp.zoo or e is sp.oo or e is sp.nan:
        raise NotImplementedError(f"from_sympy: unsupported special value {e}")

    if e.is_Add:
        return _core.add_many(cargs)
    if e.is_Mul:
        return _core.mul_many(cargs)
    if e.is_Pow:
        return cargs[0] ** cargs[1]

    # 关系算子 → 0/1 指示函数
    rel_map = {
        sp.StrictLessThan: "lt", sp.LessThan: "le",
        sp.StrictGreaterThan: "gt", sp.GreaterThan: "ge",
    }
    for cls, fname in rel_map.items():
        if isinstance(e, cls):
            return _core.make_func(fname, cargs)

    # 布尔组合（操作数已是 0/1 指示值）：And → min，Or → max，Not → 1-x
    if isinstance(e, sp.logic.boolalg.BooleanTrue):
        return _core.as_expr(1)
    if isinstance(e, sp.logic.boolalg.BooleanFalse):
        return _core.as_expr(0)
    if isinstance(e, sp.And):
        r = cargs[0]
        for c in cargs[1:]:
            r = _core.make_func("min", [r, c])
        return r
    if isinstance(e, sp.Or):
        r = cargs[0]
        for c in cargs[1:]:
            r = _core.make_func("max", [r, c])
        return r
    if isinstance(e, sp.Not):
        return 1 - cargs[0]

    # Piecewise 的分支节点：透传 (值, 条件, 原条件为 True?) 三元组
    if isinstance(e, sp.functions.elementary.piecewise.ExprCondPair):
        return (cargs[0], cargs[1], e.args[1] is sp.true)

    if isinstance(e, sp.Piecewise):
        # ((v1,c1),...,(vn,cn)) → 嵌套 where，右结合；
        # 无 True 兜底时越界处按 sympy 语义为 nan
        result = None
        for val, cond, is_true in reversed(cargs):
            if result is None and is_true:
                result = val
                continue
            fallback = result if result is not None else _core.as_expr(float("nan"))
            result = _core.make_func("where", [cond, val, fallback])
        return result

    if isinstance(e, sp.Heaviside):
        # Heaviside(x) → where(x>0, 1, where(x<0, 0, H0))；sympy 默认 H(0)=1/2
        h0 = cargs[1] if len(cargs) > 1 else _core.rational(1, 2)
        x = cargs[0]
        return _core.make_func("where", [
            _core.make_func("gt", [x, _core.as_expr(0)]), _core.as_expr(1),
            _core.make_func("where", [
                _core.make_func("lt", [x, _core.as_expr(0)]),
                _core.as_expr(0), h0])])

    if isinstance(e, sp.Derivative):
        raise NotImplementedError(
            "from_sympy: expression contains an unevaluated Derivative node. "
            "Call .doit() first, or declare symbols with real=True "
            "(sympy differentiates |x|, sign(x) etc. symbolically only under "
            "real assumptions)")

    # 实数域语义：re(x)=x，im(x)=0，conjugate(x)=x
    #（Frontier 数值域为 f64；sympy 默认符号按复数假设求导时会引入这些）
    if isinstance(e, sp.re) or isinstance(e, sp.conjugate):
        return cargs[0]
    if isinstance(e, sp.im):
        return _core.as_expr(0)

    if e.is_Function:
        name = _registry_name(type(e).__name__)
        if name in ("max", "min") and len(cargs) != 2:
            # sympy Max/Min 是 n 元：折叠为嵌套二元调用
            r = cargs[0]
            for c in cargs[1:]:
                r = _core.make_func(name, [r, c])
            return r
        if name is not None:
            return _core.make_func(name, cargs)
        raise NotImplementedError(
            f"from_sympy: function '{type(e).__name__}' is not registered in "
            f"frontier (available: {sorted(_registry_names())})")

    raise NotImplementedError(
        f"from_sympy: unsupported sympy node {type(e).__name__}: {e}")


def from_sympy(expr):
    """SymPy 表达式 → Frontier 表达式。

    迭代式后序遍历：深度不受 sys.recursionlimit 限制；
    以 sympy 节点为 key 做 memo，DAG 共享子树只转换一次。
    """
    import sympy as sp

    if isinstance(expr, _core.Expr):
        return expr

    root = sp.sympify(expr)
    # memo 以 id() 为 key：避免对深表达式触发 sympy 递归式 __hash__
    #（sympy 的对象缓存保证相同子树通常是同一对象，共享仍然生效）
    memo: dict[int, Any] = {}
    stack = [root]
    while stack:
        e = stack[-1]
        if id(e) in memo:
            stack.pop()
            continue
        pending = [a for a in e.args if id(a) not in memo]
        if pending:
            stack.extend(pending)
            continue
        stack.pop()
        memo[id(e)] = _convert_node(e, [memo[id(a)] for a in e.args], sp)
    return memo[id(root)]


def lambdify(args, exprs, modules=None, *, fastmath: bool = False,
             vecmath: bool = True, workers: int | str = 1,
             cache: bool = False, uniform=None, **ignored_kwargs):
    """``sympy.lambdify`` 的编译版替身。

    接受 SymPy 或 Frontier 的符号/表达式；``exprs`` 支持单表达式、
    list/tuple、以及 ``sympy.Matrix``（输出 reshape 回矩阵形状，
    批量输入时形状为 ``(*matrix.shape, n)``）。

    与 ``sympy.lambdify`` 的差异：输入是标量或一维数组（自动广播），
    返回 float64；首次调用前有一次 JIT 编译开销。
    """
    from frontier._compile import CompiledFunction

    # sympy.lambdify 兼容：modules/cse/docstring_limit 等参数接受并忽略
    #（编译语义下它们没有意义；modules 恒为编译内核）
    unknown = set(ignored_kwargs) - {"cse", "docstring_limit", "dummify", "printer"}
    if unknown:
        raise TypeError(f"lambdify: unsupported keyword argument(s) {sorted(unknown)}")

    # 单符号（sympy.lambdify(x, expr) 形式）自动包装
    if not isinstance(args, (list, tuple)):
        args = [args]
    arg_list = [from_sympy(a) for a in args]

    shape = None
    single = False
    try:
        import sympy as sp
        if isinstance(exprs, sp.MatrixBase):
            shape = exprs.shape
            expr_list = [from_sympy(e) for e in exprs]  # 行优先展平
        elif isinstance(exprs, (list, tuple)):
            expr_list = [from_sympy(e) for e in exprs]
        else:
            single = True
            expr_list = [from_sympy(exprs)]
    except ImportError:  # 无 sympy 环境也可用（纯 fr 表达式）
        if isinstance(exprs, (list, tuple)):
            expr_list = [from_sympy(e) for e in exprs]
        else:
            single = True
            expr_list = [from_sympy(exprs)]

    fn = CompiledFunction(expr_list, arg_list, fastmath=fastmath,
                          vecmath=vecmath, workers=workers, cache=cache,
                          uniform=[from_sympy(u) for u in uniform] if uniform else None)

    if shape is None and single:
        return _SingleWrapper(fn)
    if shape is None:
        return fn
    return _MatrixWrapper(fn, shape)


class _WrapperBase:
    """未覆盖的属性/方法（eval_scalars/eval_stacked/ir/optimized_ir/
    args/workers…）全部委托给底层 CompiledFunction。"""

    def __init__(self, fn):
        self.fn = fn

    def __getattr__(self, name):
        return getattr(self.fn, name)


class _SingleWrapper(_WrapperBase):
    """单表达式：直接返回数组（与 lambdify 单表达式行为对齐）。"""

    def __call__(self, *arrays):
        return self.fn(*arrays)[0]


class _MatrixWrapper(_WrapperBase):
    """Matrix 输出：reshape 回 (rows, cols[, n])。"""

    def __init__(self, fn, shape):
        super().__init__(fn)
        self.shape = shape

    def __call__(self, *arrays):
        stacked = self.fn.eval_stacked(*arrays).reshape(*self.shape, -1)
        if stacked.shape[-1] == 1:
            return stacked[..., 0]
        return stacked
