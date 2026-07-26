"""v3 可用性功能群：subs、hessian、min/max、JIT 磁盘缓存。"""
import numpy as np
import pytest

import frontier as fr


def test_subs_symbol():
    x, y = fr.symbols("x y")
    e = x**2 + y
    assert e.subs({x: 3}) == 9 + y
    assert e.subs({x: y}) == y**2 + y
    # 规范化随替换自动发生
    assert (x + y).subs({y: -x}).is_zero


def test_subs_subexpression():
    x, y, z = fr.symbols("x y z")
    e = fr.sin(x * y) + fr.sin(x * y) ** 2
    # 键可为任意子表达式
    r = e.subs({fr.sin(x * y): z})
    assert r == z + z**2
    # 模块级函数形式
    assert fr.subs(e, {fr.sin(x * y): z}) == z + z**2


def test_hessian():
    x, y = fr.symbols("x y")
    f = x**3 * y + y**2
    H = fr.hessian(f, [x, y])
    assert H[0][0] == 6 * x * y
    assert H[0][1] == H[1][0] == 3 * x**2
    assert H[1][1] == fr.as_expr(2)


def test_min_max_values_and_derivs():
    x, y = fr.symbols("x y")
    g = fr.compile([fr.max(x, y), fr.min(x, y)], args=(x, y))
    xs = np.array([1.0, -2.0, 3.0, 0.0])
    ys = np.array([0.5, 4.0, 3.0, -1.0])
    mx, mn = g(xs, ys)
    np.testing.assert_allclose(mx, np.maximum(xs, ys))
    np.testing.assert_allclose(mn, np.minimum(xs, ys))

    # 次梯度：max 对 a 的导数在 a>b 处为 1，a<b 处为 0
    d = fr.compile(fr.diff(fr.max(x, y), x), args=(x, y))
    np.testing.assert_allclose(d(np.array([2.0, 0.0]), np.array([1.0, 1.0])),
                               [1.0, 0.0])


def test_max_via_sympy():
    sp = pytest.importorskip("sympy")
    x, y = sp.symbols("x y")
    e = sp.Max(x, y)  # sympy 的 Max 类名 → 注册表 max
    try:
        f = fr.from_sympy(e)
    except NotImplementedError:
        pytest.skip("Max mapping not enabled")
    g = fr.compile(f, args=[fr.symbol("x"), fr.symbol("y")])
    assert g(np.array([1.0]), np.array([2.0]))[0] == 2.0


def test_uniform_args():
    """uniform 参数：循环外载入 + 参数子表达式 LICM 提升。"""
    a, b, x = fr.symbols("a b x")
    e = a / (b + 1) * fr.exp(-a * x)
    g = fr.compile(e, args=(a, b, x), uniform=(a, b))
    xs = np.linspace(0, 1, 1001)
    want = 2.0 / (3.0 + 1) * np.exp(-2.0 * xs)
    np.testing.assert_allclose(g(2.0, 3.0, xs), want, rtol=1e-14)
    # eval_stacked 同样支持
    np.testing.assert_allclose(
        fr.compile([e, e * 2], args=(a, b, x), uniform=(a, b))
        .eval_stacked(2.0, 3.0, xs)[1],
        2 * want, rtol=1e-14)
    # uniform 实参传数组应报错
    with pytest.raises(ValueError):
        g(np.array([1.0, 2.0]), 3.0, xs)


def test_jit_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTIER_CACHE_DIR", str(tmp_path))
    from frontier import _jit
    _jit.jit_compile.cache_clear()  # 绕过进程内 LRU，强制走磁盘路径

    x, y = fr.symbols("x y")
    e = fr.sin(x) * fr.exp(y) + x**3

    g1 = fr.compile(e, args=(x, y), cache=True)
    assert not g1._kernel.cache_hit
    objs = list(tmp_path.glob("*.o"))
    assert len(objs) == 1 and objs[0].stat().st_size > 0

    _jit.jit_compile.cache_clear()
    g2 = fr.compile(e, args=(x, y), cache=True)
    assert g2._kernel.cache_hit  # 第二次：磁盘命中，跳过优化

    xs = np.linspace(0, 1, 11)
    np.testing.assert_allclose(g1(xs, xs), g2(xs, xs))
