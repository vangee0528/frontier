"""v0.4/v0.5 功能回归：where/比较/erf、symbols range、适配器外的 API 补全。"""
import numpy as np
import pytest

import frontier as fr


def test_where_and_comparisons():
    x = fr.symbols("x")
    xs = np.linspace(-2, 2, 401)
    g = fr.compile(fr.where(x > 0, x, -x), args=(x,))
    np.testing.assert_allclose(g(xs), np.abs(xs))

    d = fr.compile(fr.diff(fr.where(x > 0, x**2, x), x), args=(x,))
    np.testing.assert_allclose(d(xs), np.where(xs > 0, 2 * xs, 1.0))

    for op, ref in [(x < 1, xs < 1), (x <= 0, xs <= 0),
                    (x > -1, xs > -1), (x >= 0.5, xs >= 0.5)]:
        gg = fr.compile(op, args=(x,))
        np.testing.assert_allclose(gg(xs), ref.astype(float))


def test_erf():
    scipy_special = pytest.importorskip("scipy.special")
    sperf, sperfc = scipy_special.erf, scipy_special.erfc
    x = fr.symbols("x")
    xs = np.linspace(-3, 3, 601)
    g = fr.compile([fr.erf(x), fr.erfc(x), fr.diff(fr.erf(x), x)], args=(x,))
    v, vc, dv = g(xs)
    np.testing.assert_allclose(v, sperf(xs), rtol=1e-12)
    np.testing.assert_allclose(vc, sperfc(xs), rtol=1e-12)
    np.testing.assert_allclose(dv, 2 / np.sqrt(np.pi) * np.exp(-xs**2), rtol=1e-12)


def test_sympy_piecewise_heaviside():
    sp = pytest.importorskip("sympy")
    sx = sp.Symbol("x")
    xs = np.linspace(-2, 2, 401)

    pw = sp.Piecewise((0, sx < -1), (sx + 1, sx < 1), (2, True))
    f = fr.lambdify([sx], pw)
    np.testing.assert_allclose(
        f(xs), np.where(xs < -1, 0, np.where(xs < 1, xs + 1, 2)))

    h = fr.lambdify([sx], sp.Heaviside(sx))
    np.testing.assert_allclose(h(np.array([-1.0, 0.0, 1.0])), [0, 0.5, 1])

    b = fr.lambdify([sx], sp.Piecewise((1, sp.And(sx > 0, sx < 1)), (0, True)))
    np.testing.assert_allclose(b(xs), ((xs > 0) & (xs < 1)).astype(float))


def test_symbols_range_syntax():
    q = fr.symbols("q:3")
    assert [str(s) for s in q] == ["q0", "q1", "q2"]
    letters = fr.symbols("a:d")
    assert [str(s) for s in letters] == ["a", "b", "c", "d"]
    mixed = fr.symbols("x y:2")
    assert [str(s) for s in mixed] == ["x", "y0", "y1"]
    with pytest.raises(ValueError):
        fr.symbols("x:")


def test_lambdify_sympy_compat_signature():
    sp = pytest.importorskip("sympy")
    sx = sp.Symbol("x")
    f = fr.lambdify(sx, sx**2, modules="numpy", cse=True)  # 单符号 + 兼容 kwargs
    np.testing.assert_allclose(f(np.array([3.0])), [9.0])
    with pytest.raises(TypeError):
        fr.lambdify([sx], sx, not_a_real_kwarg=1)


def test_api_completeness():
    x, a, b = fr.symbols("x a b")
    e = fr.sin(x) * a + b

    assert {str(s) for s in e.free_symbols} == {"x", "a", "b"}
    assert float(fr.as_expr(2.5)) == 2.5
    with pytest.raises(TypeError):
        float(x)
    assert (x**2).diff(x) == 2 * x
    assert abs(float(fr.pi) - np.pi) < 1e-15

    # 缺失符号报错列出全部缺失 + args 清单
    with pytest.raises(fr.CompileError, match=r"'a'.*'b'|'b'.*'a'"):
        fr.compile(e, args=(x,))

    # 零参编译
    assert fr.compile(fr.as_expr(7), args=())()[0] == 7.0

    # 复数拒绝
    g = fr.compile(x + a, args=(x, a))
    with pytest.raises(TypeError, match="complex"):
        g(np.ones(3, dtype=complex), np.ones(3))

    # uniform 写错符号即报错
    with pytest.raises(ValueError, match="uniform"):
        fr.compile(x + a, args=(x, a), uniform=(b,))


def test_uniform_with_workers():
    a, b, x = fr.symbols("a b x")
    xs = np.linspace(0, 1, 200_000)
    g = fr.compile(a * x + b, args=(a, b, x), uniform=(a, b), workers=4)
    np.testing.assert_allclose(g(2.0, 3.0, xs), 2 * xs + 3)


def test_eval_scalars_safety():
    import threading
    x, a = fr.symbols("x a")
    f = fr.compile([x * a, x + a], args=(x, a))
    with pytest.raises(TypeError):
        f.eval_scalars(1.0)
    errs = []

    def worker(v):
        for _ in range(500):
            out = f.eval_scalars(v, 10.0)
            if out[0] != v * 10 or out[1] != v + 10:
                errs.append(out)

    ts = [threading.Thread(target=worker, args=(float(i + 1),)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs


def test_dangling_expression_no_crash_at_exit():
    """进程退出时悬挂 Add(Mul) 表达式不再段错误（回归：CRT 收尾 UB）。"""
    import subprocess, sys
    from pathlib import Path
    code = ("import frontier as fr\n"
            "x, a, b = fr.symbols('x a b')\n"
            "e = x*a + b\n"
            "print('ok')\n")
    env_path = str(Path(__file__).resolve().parent.parent / "python")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=60,
                       env={**__import__('os').environ, "PYTHONPATH": env_path})
    assert r.returncode == 0 and "ok" in r.stdout
