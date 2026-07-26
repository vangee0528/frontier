"""fr.compile() 编译管线测试。"""
import numpy as np
import pytest

import frontier as fr


@pytest.fixture
def xy():
    return fr.symbols("x y")


def test_single_output(xy):
    x, y = xy
    g = fr.compile(fr.sin(x * y) + x**2, args=(x, y))
    xs = np.linspace(-2, 2, 101)
    ys = np.linspace(1, 3, 101)
    assert np.allclose(g(xs, ys), np.sin(xs * ys) + xs**2)


def test_multi_output_tuple(xy):
    x, y = xy
    g = fr.compile([x + y, x * y, x - y], args=(x, y))
    xs = np.array([1.0, 2.0])
    ys = np.array([3.0, 4.0])
    a, b, c = g(xs, ys)
    assert np.allclose(a, xs + ys)
    assert np.allclose(b, xs * ys)
    assert np.allclose(c, xs - ys)


def test_gradient_pipeline(xy):
    x, y = xy
    f = fr.exp(-(x**2) - y**2) * fr.sin(x)
    g = fr.compile(fr.grad(f, [x, y]), args=(x, y))
    xs = np.linspace(-1, 1, 51)
    ys = np.linspace(-1, 1, 51)
    gx, gy = g(xs, ys)

    def fnum(a, b):
        return np.exp(-a**2 - b**2) * np.sin(a)

    eps = 1e-7
    assert np.allclose(gx, (fnum(xs + eps, ys) - fnum(xs, ys)) / eps, atol=1e-5)
    assert np.allclose(gy, (fnum(xs, ys + eps) - fnum(xs, ys)) / eps, atol=1e-5)


def test_pow_specializations():
    x = fr.symbols("x")
    g = fr.compile(
        [x ** fr.rational(1, 2), 1 / x, x**3, x ** fr.rational(-1, 2), x**2.5],
        args=(x,),
    )
    xs = np.array([0.25, 1.0, 4.0])
    sqrt, inv, cube, rsqrt, p25 = g(xs)
    assert np.allclose(sqrt, np.sqrt(xs))
    assert np.allclose(inv, 1 / xs)
    assert np.allclose(cube, xs**3)
    assert np.allclose(rsqrt, 1 / np.sqrt(xs))
    assert np.allclose(p25, xs**2.5)


def test_scalar_broadcast(xy):
    x, y = xy
    g = fr.compile(x + y, args=(x, y))
    ys = np.array([1.0, 2.0, 3.0])
    assert np.allclose(g(10.0, ys), 10.0 + ys)
    # 全标量 → 长度 1
    assert g(1.0, 2.0).shape == (1,)


def test_constant_expression(xy):
    x, y = xy
    g = fr.compile(fr.as_expr(7), args=(x,))
    assert np.allclose(g(np.array([1.0, 2.0])), 7.0)


def test_free_symbol_error(xy):
    x, y = xy
    with pytest.raises(fr.CompileError):
        fr.compile(x + y, args=(x,))


def test_input_validation(xy):
    x, y = xy
    g = fr.compile(x + y, args=(x, y))
    with pytest.raises(TypeError):
        g(np.array([1.0]))  # 少一个输入
    with pytest.raises(ValueError):
        g(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))  # 长度不匹配


def test_fastmath_close_enough(xy):
    x, y = xy
    f = fr.sin(x) * fr.cos(y) + fr.exp(x * 0.1)
    g0 = fr.compile(f, args=(x, y))
    g1 = fr.compile(f, args=(x, y), fastmath=True)
    xs = np.linspace(-3, 3, 101)
    ys = np.linspace(-3, 3, 101)
    assert np.allclose(g0(xs, ys), g1(xs, ys), rtol=1e-10)


def test_jit_cache(xy):
    x, y = xy
    a = fr.compile(x * y, args=(x, y))
    b = fr.compile(x * y, args=(x, y))
    # 相同 IR 命中同一 JIT 内核
    assert a._kernel is b._kernel


def test_cse_in_ir(xy):
    """共享子表达式在 IR 中只出现一次（CSE 经 hash-consing 天然完成）。"""
    x, y = xy
    u = fr.sin(x * y)
    g = fr.compile([u + u**2, u * 2], args=(x, y))
    assert g.ir.count("call double @fr_sin") == 1
