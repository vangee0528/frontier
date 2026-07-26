"""与 SymPy 的交叉验证：同一表达式两边求导、两边求值，数值必须一致。

Frontier 的求导/化简正确性以 SymPy 为参照系（独立实现，互为对照）。
"""
import numpy as np
import pytest

sympy = pytest.importorskip("sympy")

import frontier as fr  # noqa: E402

# 每个用例：(表达式构造器 fr, 表达式构造器 sympy, 变量个数)
# 两边用同构的写法分别构造，避免任何共享代码路径。
CASES = [
    ("poly", lambda s, m: 3 * m[0] ** 4 - 2 * m[0] ** 2 + m[0] - 7, 1),
    ("rational_pow",
     lambda s, m: m[0] ** s.Rational(3, 2) + m[0] ** s.Rational(-1, 2), 1),
    ("trig", lambda s, m: s.sin(m[0]) * s.cos(m[1]) + s.tan(m[0] * m[1]), 2),
    ("exp_log", lambda s, m: s.exp(-m[0] ** 2) * s.log(m[1] + 3), 2),
    ("nested", lambda s, m: s.sin(s.exp(m[0]) + s.cos(m[1]) ** 2), 2),
    ("gauss2d", lambda s, m: s.exp(-(m[0] ** 2 + m[1] ** 2) / 2) * s.sin(m[0] * m[1]), 2),
    ("hyper", lambda s, m: s.sinh(m[0]) * s.tanh(m[1]) + s.cosh(m[0] * m[1] / 5), 2),
    ("inv_trig", lambda s, m: s.atan(m[0]) + s.asin(m[0] / 3), 1),
    ("quotient", lambda s, m: (m[0] + m[1]) / (m[0] ** 2 + m[1] ** 2 + 1), 2),
    ("power_tower", lambda s, m: (m[0] ** 2 + 1) ** (m[1] / 7), 2),
]


class FrNS:
    """让同一个 lambda 既能吃 sympy 命名空间也能吃 frontier 命名空间。"""

    Rational = staticmethod(fr.rational)

    def __getattr__(self, name):
        return getattr(fr, name)


@pytest.mark.parametrize("name,build,nvars", CASES, ids=[c[0] for c in CASES])
def test_value_and_gradient_match_sympy(name, build, nvars):
    rng = np.random.default_rng(42)
    # 采样点避开奇点：取 (0.2, 1.4) 区间
    samples = [rng.uniform(0.2, 1.4, size=200) for _ in range(nvars)]

    # frontier 侧
    fvars = [fr.symbol(f"v{i}") for i in range(nvars)]
    fexpr = build(FrNS(), fvars)
    fgrad = fr.grad(fexpr, fvars)
    fn = fr.compile([fexpr, *fgrad], args=fvars)
    fr_results = fn(*samples)

    # sympy 侧
    svars = [sympy.Symbol(f"v{i}") for i in range(nvars)]
    sexpr = build(sympy, svars)
    sgrad = [sympy.diff(sexpr, v) for v in svars]
    sfn = sympy.lambdify(svars, [sexpr, *sgrad], modules="numpy")
    sp_results = sfn(*samples)

    for got, want, label in zip(
            fr_results, sp_results,
            ["value"] + [f"d/dv{i}" for i in range(nvars)]):
        np.testing.assert_allclose(
            got, np.broadcast_to(want, got.shape), rtol=1e-9, atol=1e-12,
            err_msg=f"{name}: {label} mismatch vs sympy")
