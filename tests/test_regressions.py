"""历史缺陷的独立回归测试。

每个用例对应一个曾在开发/模糊测试中发现并修复的真实缺陷；
fuzzer 不再对这些类别做任何跳过——此文件是它们不复发的保证。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import frontier as fr


class TestPowOfMulCrash:
    """曾触发规范形断言 abort：Mul 因子的 base 为 Mul/Pow（符号指数）。

    历史最小复现（fuzz 发现）：2*((x*y)**z)、-((x**(3/2))**y)、
    fr.diff((x**2)**y, x)。修复：放宽过严的因子不变量，
    符号指数下 (x·y)^z / (x^y)^z 合法保持嵌套。
    """

    def test_construction(self):
        x, y, z = fr.symbols("x y z")
        e1 = 2 * ((x * y) ** z)
        e2 = -((x ** fr.rational(3, 2)) ** y)
        e3 = (x * y) ** z * (x * y) ** z          # 同底合并 → 指数翻倍
        assert "x*y" in repr(e1)
        assert e3 == (x * y) ** (2 * z)

    def test_diff(self):
        x, y = fr.symbols("x y")
        d = fr.diff((x**2) ** y, x)               # 曾直接 abort
        g = fr.compile(d, args=(x, y))
        xs = np.array([1.3]); ys = np.array([0.7])
        # d/dx (x²)^y = 2y·x^(2y-1)
        want = 2 * 0.7 * 1.3 ** (2 * 0.7 - 1)
        np.testing.assert_allclose(g(xs, ys)[0], want, rtol=1e-12)

    def test_compile_and_eval(self):
        x, y, z = fr.symbols("x y z")
        g = fr.compile(2 * ((x * y) ** z), args=(x, y, z))
        out = g(np.array([1.3]), np.array([0.7]), np.array([2.1]))
        np.testing.assert_allclose(out[0], 2 * (1.3 * 0.7) ** 2.1, rtol=1e-12)


def test_nary_max_min_from_sympy():
    """sympy Max/Min 为 n 元，曾直接报'expects 2 arguments'。"""
    sp = pytest.importorskip("sympy")
    x, y = sp.symbols("x y", real=True)
    f = fr.lambdify([x, y], sp.Min(x, x**2, y) + sp.Max(x, y, 3))
    xs = np.array([0.5, 2.0, -1.0])
    ys = np.array([1.0, 0.1, 4.0])
    want = np.minimum(np.minimum(xs, xs**2), ys) + np.maximum(np.maximum(xs, ys), 3)
    np.testing.assert_allclose(f(xs, ys), want)


def test_deep_expression_construct_and_teardown():
    """深链表达式（数千层）构造与析构曾栈溢出（shared_ptr 递归释放）。"""
    sp = pytest.importorskip("sympy")
    x = sp.Symbol("x")
    e = x
    for _ in range(1500):
        e = sp.sin(e, evaluate=False)
    fe = fr.from_sympy(e)   # 迭代式转换
    del fe                  # 迭代式析构


def test_exit_with_dangling_expression():
    """进程退出时悬挂 Add(Mul) 表达式曾段错误（CRT 收尾 TLS UB）。"""
    code = ("import frontier as fr\n"
            "x, a, b = fr.symbols('x a b')\n"
            "e = x*a + b\n"
            "print('ok')\n")
    py_dir = Path(fr.__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=60,
                       env={**__import__('os').environ,
                            "PYTHONPATH": str(py_dir)})
    assert r.returncode == 0 and "ok" in r.stdout


def test_exp_subnormal_range():
    """exp 在次正规结果区间曾输出垃圾值（2^n 位构造指数越界）。"""
    x = fr.symbols("x")
    g = fr.compile(fr.exp(x), args=(x,))
    xs = np.linspace(-745.0, -700.0, 20001)
    got = g(xs)
    want = np.exp(xs)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, want, atol=5e-320)


def test_multiple_kernels_teardown():
    """多个 JIT 内核共存与释放曾因 TargetMachine 共享 use-after-free。"""
    x = fr.symbols("x")
    kernels = [fr.compile(x + i, args=(x,)) for i in range(8)]
    xs = np.array([1.0])
    for i, k in enumerate(kernels):
        np.testing.assert_allclose(k(xs), [1.0 + i])
    del kernels  # 逐个析构引擎不应崩溃


def test_uniform_with_workers_no_oob():
    """uniform×workers 并行分块曾对单元素缓冲加块偏移（越界读）。"""
    a, b, x = fr.symbols("a b x")
    xs = np.linspace(0, 1, 200_000)
    g = fr.compile(a * x + b, args=(a, b, x), uniform=(a, b), workers=4)
    np.testing.assert_allclose(g(2.0, 3.0, xs), 2 * xs + 3)

def test_merged_integer_exponent_renormalizes():
    """CI fuzz（seed 1101 idx 175）发现：同底合并出整数指数后未回流
    pow() 分配展开，(-x)^(-1) 丢符号 → 导数整体反号。"""
    sp = pytest.importorskip("sympy")
    x1 = sp.Symbol("x1", real=True)
    e = sp.log((-x1) ** sp.Rational(3, 2))
    fx = fr.symbol("x1")
    d = fr.diff(fr.from_sympy(e), fx)
    got = fr.compile(d, args=(fx,))(np.array([-1.3]))[0]
    want = float(sp.diff(e, x1).subs(x1, -1.3))          # 3/(2x)
    assert abs(got - want) < 1e-12

    # 结构层面：合并出的整数指数因子必须已展开
    x = fr.symbols("x")
    nx = -x
    assert nx ** fr.rational(1, 2) * nx ** fr.rational(-3, 2) == -(x ** -1)
