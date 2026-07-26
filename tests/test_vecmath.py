"""向量化数学函数（fr_sin/fr_cos/fr_exp/fr_log）精度与边界回归。"""
import numpy as np
import pytest

import frontier as fr


@pytest.fixture(scope="module")
def x():
    return fr.symbols("x")


def test_sin_cos_accuracy(x):
    xs = np.concatenate([
        np.linspace(-100, 100, 100001),
        np.linspace(-1e6, 1e6, 10001),
        np.array([0.0, np.pi / 2, np.pi, 1e-300, -1e-300]),
    ])
    g = fr.compile([fr.sin(x), fr.cos(x)], args=(x,))
    s, c = g(xs)
    assert np.max(np.abs(s - np.sin(xs))) < 5e-16
    assert np.max(np.abs(c - np.cos(xs))) < 5e-16


def test_exp_accuracy_and_bounds(x):
    xs = np.linspace(-700, 700, 100001)
    g = fr.compile(fr.exp(x), args=(x,))
    got = g(xs)
    rel = np.abs(got - np.exp(xs)) / np.exp(xs)
    assert np.max(rel) < 1e-15

    edge = g(np.array([710.0, 1e300, -800.0, -1e300, np.nan, 0.0]))
    assert edge[0] == np.inf and edge[1] == np.inf
    assert edge[2] == 0.0 and edge[3] == 0.0
    assert np.isnan(edge[4]) and edge[5] == 1.0


def test_log_accuracy_and_bounds(x):
    xs = np.concatenate([
        np.geomspace(1e-300, 1e300, 100001),
        np.array([1.0, np.e]),
    ])
    g = fr.compile(fr.log(x), args=(x,))
    got = g(xs)
    want = np.log(xs)
    rel = np.abs(got - want) / np.maximum(np.abs(want), 1.0)
    assert np.max(rel) < 1e-15

    edge = g(np.array([0.0, -1.0, np.inf, np.nan]))
    assert edge[0] == -np.inf
    assert np.isnan(edge[1])
    assert edge[2] == np.inf and np.isnan(edge[3])


def test_vecmath_enables_simd(x):
    """优化后 IR 应出现向量类型（宿主 CPU 支持 SIMD 时）。"""
    import re
    g = fr.compile(fr.sin(x * x) + fr.exp(x), args=(x,))
    assert re.search(r"<\d+ x double>", g.optimized_ir), "loop not vectorized"


def test_vecmath_off_uses_libm(x):
    g = fr.compile(fr.sin(x), args=(x,), vecmath=False)
    assert "llvm.sin.f64" in g.ir
    assert "fr_sin" not in g.ir
    # 两条路径数值一致（1-2 ulp 内）
    xs = np.linspace(-10, 10, 1001)
    g2 = fr.compile(fr.sin(x), args=(x,), vecmath=True)
    np.testing.assert_allclose(g(xs), g2(xs), rtol=0, atol=5e-16)


def test_eval_stacked(x):
    y = fr.symbols("y")
    g = fr.compile([x + y, x * y, x - y], args=(x, y))
    xs = np.array([1.0, 2.0, 3.0])
    ys = np.array([4.0, 5.0, 6.0])
    out = g.eval_stacked(xs, ys)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, np.stack(g(xs, ys)))
