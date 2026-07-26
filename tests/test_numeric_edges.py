"""数值边界行为：IEEE 特殊值、极端参数、不可导点、语义约定。

这些测试固化 Frontier 的**文档承诺**（docs/performance.md 已知限制），
不是宽泛的"越准越好"——如 sin 大参数的精度渐降是声明过的行为。
"""
import numpy as np
import pytest

import frontier as fr


@pytest.fixture(scope="module")
def x():
    return fr.symbols("x")


class TestIEEESpecials:
    def test_nan_propagation(self, x):
        g = fr.compile([fr.sin(x), fr.exp(x), x**2, fr.where(x > 0, x, -x)],
                       args=(x,))
        outs = g(np.array([np.nan]))
        for o in outs[:3]:
            assert np.isnan(o[0])
        # where 的 NaN 条件走 false 分支（fcmp one 语义），结果 -NaN 仍 NaN
        assert np.isnan(outs[3][0])

    def test_inf_arithmetic(self, x):
        g = fr.compile([fr.exp(x), fr.exp(-x), x + 1, 1 / x], args=(x,))
        e, en, p, inv = g(np.array([np.inf]))
        assert e[0] == np.inf and en[0] == 0.0 and p[0] == np.inf
        assert inv[0] == 0.0

    def test_negative_zero(self, x):
        g = fr.compile([1 / x, fr.sign(x), fr.abs(x)], args=(x,))
        inv, sgn, ab = g(np.array([-0.0]))
        assert inv[0] == -np.inf           # 1/-0 = -inf（IEEE）
        assert sgn[0] == 0.0               # sign(-0) = 0（与 numpy 一致）
        assert ab[0] == 0.0

    def test_subnormal_inputs_and_outputs(self, x):
        tiny = 5e-310                       # 次正规
        g = fr.compile([x * 2, x + x, fr.sqrt(x * x)], args=(x,))
        d, s, r = g(np.array([tiny]))
        assert d[0] == 2 * tiny and s[0] == 2 * tiny
        # exp 的次正规输出（独立回归另见 test_regressions）
        ge = fr.compile(fr.exp(x), args=(x,))
        np.testing.assert_allclose(ge(np.array([-745.0])),
                                   np.exp(np.array([-745.0])), atol=5e-320)


class TestExtremeArguments:
    def test_trig_large_args_documented_domain(self, x):
        """|x| ≤ 1e8：精度完整（文档承诺域）。"""
        xs = np.array([1e6, 3.6e7, 9.9e7, -8.7e7])
        g = fr.compile([fr.sin(x), fr.cos(x)], args=(x,))
        s, c = g(xs)
        np.testing.assert_allclose(s, np.sin(xs), atol=2e-9)  # 规约后绝对界
        np.testing.assert_allclose(c, np.cos(xs), atol=2e-9)

    def test_trig_huge_args_finite_and_bounded(self, x):
        """|x| > 1e8：文档声明精度渐降，但必须有界且非 NaN。"""
        xs = np.array([1e10, 1e15, -3e12])
        g = fr.compile([fr.sin(x), fr.cos(x)], args=(x,))
        s, c = g(xs)
        assert np.all(np.isfinite(s)) and np.all(np.abs(s) <= 1.0 + 1e-12)
        assert np.all(np.isfinite(c)) and np.all(np.abs(c) <= 1.0 + 1e-12)

    def test_trig_libm_fallback_exact(self, x):
        """vecmath=False 时任意大参数与 libm 完全一致。"""
        xs = np.array([1e10, 1e15, 1e18])
        g = fr.compile(fr.sin(x), args=(x,), vecmath=False)
        np.testing.assert_array_equal(g(xs), np.sin(xs))


class TestPowSemantics:
    def test_negative_base_fractional_exponent_is_nan(self, x):
        """实数域语义：(-8)^(1/3) 等负底分数幂 = NaN（文档承诺；
        sympy 走复数主分支会给出复数结果，属声明过的差异）。"""
        y = fr.symbols("y")
        g = fr.compile(x**y, args=(x, y))
        out = g(np.array([-8.0, -2.0]), np.array([1 / 3, 0.5]))
        assert np.all(np.isnan(out))

    def test_negative_base_integer_exponent_exact(self, x):
        g = fr.compile([x**3, x**(-2), x**4], args=(x,))
        a, b, c = g(np.array([-2.0]))
        assert a[0] == -8.0 and b[0] == 0.25 and c[0] == 16.0

    def test_zero_to_zero_is_one(self, x):
        """0^0 = 1：符号层构造期约定（与 numpy/IEEE pow 一致）。"""
        y = fr.symbols("y")
        g = fr.compile(x**y, args=(x, y))
        assert g(np.array([0.0]), np.array([0.0]))[0] == 1.0

    def test_zero_to_negative_is_inf(self, x):
        y = fr.symbols("y")
        g = fr.compile(x**y, args=(x, y))
        assert g(np.array([0.0]), np.array([-1.0]))[0] == np.inf


class TestPiecewiseBoundaries:
    def test_where_exact_boundary(self, x):
        """边界点 x==0：x>0 为假 → false 分支（与 numpy.where 一致）。"""
        g = fr.compile(fr.where(x > 0, 1.0, -1.0), args=(x,))
        out = g(np.array([-1e-300, -0.0, 0.0, 1e-300]))
        np.testing.assert_array_equal(out, [-1, -1, -1, 1])

    def test_comparison_near_boundary(self, x):
        g = fr.compile([x >= 1, x < 1], args=(x,))
        eps = np.finfo(float).eps
        ge, lt = g(np.array([1 - eps, 1.0, 1 + eps]))
        np.testing.assert_array_equal(ge, [0, 1, 1])
        np.testing.assert_array_equal(lt, [1, 0, 0])


class TestNonDifferentiablePoints:
    """不可导点的导数语义：几乎处处正确的次梯度约定（文档化行为）。"""

    def test_abs_at_zero(self, x):
        d = fr.compile(fr.diff(fr.abs(x), x), args=(x,))
        # d|x|/dx = sign(x)：x=0 处返回 0（次梯度选择，与 sympy 一致）
        np.testing.assert_array_equal(d(np.array([-2.0, 0.0, 3.0])),
                                      [-1.0, 0.0, 1.0])

    def test_max_at_tie(self, x):
        y = fr.symbols("y")
        d = fr.compile(fr.diff(fr.max(x, y), x), args=(x, y))
        # a==b 平局点：∂max/∂a = (1+sign(0))/2 = 1/2（对称次梯度）
        assert d(np.array([1.0]), np.array([1.0]))[0] == 0.5

    def test_where_branch_derivative_at_boundary(self, x):
        d = fr.compile(fr.diff(fr.where(x > 0, x**2, -x), x), args=(x,))
        # 边界 x=0 取 false 分支导数（-1）；跳变点 δ 贡献按文档忽略
        np.testing.assert_array_equal(d(np.array([-1.0, 0.0, 1.0])),
                                      [-1.0, -1.0, 2.0])

    def test_sqrt_at_zero(self, x):
        d = fr.compile(fr.diff(fr.sqrt(x), x), args=(x,))
        assert d(np.array([0.0]))[0] == np.inf   # 单侧导数发散 → inf


class TestOutParameter:
    def test_eval_stacked_out(self):
        x, y = fr.symbols("x y")
        g = fr.compile([x + y, x * y], args=(x, y))
        xs = np.linspace(0, 1, 101)
        buf = np.empty((2, 101))
        r = g.eval_stacked(xs, xs, out=buf)
        assert r is buf
        np.testing.assert_allclose(buf[0], 2 * xs)
        with pytest.raises(ValueError, match="out="):
            g.eval_stacked(xs, xs, out=np.empty((3, 101)))
        with pytest.raises(ValueError, match="out="):
            g.eval_stacked(xs, xs, out=np.empty((2, 101), dtype=np.float32))

    def test_eval_scalars_out(self):
        x, y = fr.symbols("x y")
        g = fr.compile([x + y, x * y], args=(x, y))
        buf = np.empty(2)
        r = g.eval_scalars(3.0, 4.0, out=buf)
        assert r is buf
        np.testing.assert_allclose(buf, [7.0, 12.0])
        # 首次调用即带 out（曾有局部变量遮蔽参数的隐患）
        g2 = fr.compile([x - y], args=(x, y))
        buf2 = np.empty(1)
        assert g2.eval_scalars(5.0, 2.0, out=buf2) is buf2
        assert buf2[0] == 3.0
