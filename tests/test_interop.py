"""SymPy 互操作（from_sympy / lambdify）与单点快路径测试。"""
import numpy as np
import pytest

sympy = pytest.importorskip("sympy")
import sympy as sp  # noqa: E402

import frontier as fr  # noqa: E402


def test_from_sympy_roundtrip_values():
    x, y = sp.symbols("x y")
    cases = [
        x**2 + 3 * x - 1,
        sp.Rational(2, 7) * x + sp.Rational(1, 2),
        sp.sin(x) * sp.cos(y) + sp.tan(x * y),
        sp.exp(-(x**2)) * sp.log(y + 2),
        sp.sqrt(x) + x ** sp.Rational(-3, 2),
        sp.atan2(y, x) + sp.Abs(x) + sp.sign(y),
        sp.pi * x + sp.E * y,
    ]
    xs = np.linspace(0.3, 1.3, 37)
    ys = np.linspace(0.4, 1.1, 37)
    for e in cases:
        got = fr.lambdify([x, y], e)(xs, ys)
        want = sp.lambdify([x, y], e, modules="numpy")(xs, ys)
        np.testing.assert_allclose(got, np.broadcast_to(want, got.shape),
                                   rtol=1e-12, err_msg=str(e))


def test_from_sympy_exactness():
    x = sp.Symbol("x")
    # 有理数保持精确
    assert repr(fr.from_sympy(sp.Rational(1, 3) * x)) == "(1/3)*x"
    # sympy sqrt 是 Pow(x, 1/2)，转换后与 frontier 幂合并
    e = fr.from_sympy(sp.sqrt(x)) * fr.from_sympy(x ** sp.Rational(-1, 2))
    assert e.is_one


def test_from_sympy_wide_add():
    # 宽 Add/Mul 走 n 元一次规范化（不会退化为 O(k²) 两两合并）
    xs = sp.symbols("a:z")
    e = sum(xs) + sum(i * v for i, v in enumerate(xs))
    f = fr.from_sympy(e)
    assert fr.from_sympy(e) == f  # 幂等且 interning 命中


def test_from_sympy_deep_expression_no_recursion_error():
    x = sp.Symbol("x")
    e = x
    # evaluate=False 绕开 sympy 假设引擎自身的递归限制，
    # 专测转换器：1500 层嵌套远超默认递归深度，迭代式遍历不受影响
    for _ in range(1500):
        e = sp.sin(e, evaluate=False)
    fr.from_sympy(e)  # 不抛 RecursionError 即通过


def test_from_sympy_unsupported():
    x = sp.Symbol("x")
    with pytest.raises(NotImplementedError):
        fr.from_sympy(sp.gamma(x))
    with pytest.raises(NotImplementedError):
        fr.from_sympy(sp.oo * x)


def test_lambdify_matrix_shape():
    x, y = sp.symbols("x y")
    M = sp.Matrix([[x + y, x * y], [sp.sin(x), sp.cos(y)]])
    g = fr.lambdify([x, y], M)
    out = g(0.5, 0.3)
    assert out.shape == (2, 2)
    want = np.array(sp.lambdify([x, y], M)(0.5, 0.3), dtype=float)
    np.testing.assert_allclose(out, want, rtol=1e-12)

    xs = np.array([0.5, 1.0, 2.0])
    ys = np.array([0.3, 0.4, 0.5])
    assert g(xs, ys).shape == (2, 2, 3)


def test_eval_scalars_fast_path():
    x, y = fr.symbols("x y")
    f = fr.compile([x * y, x + y, x - y], args=(x, y))
    out = f.eval_scalars(3.0, 2.0)
    np.testing.assert_allclose(out, [6.0, 5.0, 1.0])
    # 与常规路径一致
    a, b, c = f(np.array([3.0]), np.array([2.0]))
    np.testing.assert_allclose(out, [a[0], b[0], c[0]])
