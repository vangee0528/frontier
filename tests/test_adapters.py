"""scipy 适配器（compile_ode / compile_objective / compile_fit）集成测试。"""
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import least_squares, minimize  # noqa: E402

import frontier as fr  # noqa: E402


def test_compile_ode_harmonic_oscillator():
    x, v, k = fr.symbols("x v k")
    f = fr.compile_ode([v, -k * x], state=[x, v], params=[k])

    sol = solve_ivp(f.rhs, (0, 2 * np.pi), [1.0, 0.0], jac=f.jac,
                    args=(1.0,), method="Radau", rtol=1e-9, atol=1e-12)
    assert sol.success
    # 周期 2π 回到起点
    np.testing.assert_allclose(sol.y[:, -1], [1.0, 0.0], atol=1e-6)

    # jac 形状与值
    J = f.jac(0.0, [1.0, 0.0], 2.0)
    np.testing.assert_allclose(J, [[0, 1], [-2, 0]])


def test_compile_ode_from_sympy_exprs():
    sp = pytest.importorskip("sympy")
    y1, y2 = sp.symbols("y1 y2")
    # 刚性 Robertson 问题的前两维简化
    f = fr.compile_ode([-0.04 * y1 + 1e4 * y2, 0.04 * y1 - 1e4 * y2],
                       state=[y1, y2])
    rhs, jac = f
    np.testing.assert_allclose(rhs(0, [1.0, 0.0]), [-0.04, 0.04])
    np.testing.assert_allclose(jac(0, [1.0, 0.0]),
                               [[-0.04, 1e4], [0.04, -1e4]])


def test_compile_objective_rosenbrock():
    x, y = fr.symbols("x y")
    rosen = (1 - x) ** 2 + 100 * (y - x**2) ** 2
    f = fr.compile_objective(rosen, [x, y])

    assert f([1.0, 1.0]) == 0.0
    np.testing.assert_allclose(f.grad([1.0, 1.0]), [0.0, 0.0], atol=1e-12)

    r = minimize(f, [-1.2, 1.0], jac=f.grad, hess=f.hess,
                 method="trust-ncg", options={"gtol": 1e-10})
    assert r.success
    np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-6)


def test_compile_objective_with_params():
    x, y, a = fr.symbols("x y a")
    f = fr.compile_objective((x - a) ** 2 + (y + a) ** 2, [x, y], params=[a])
    r = minimize(f, [0.0, 0.0], args=(3.0,), jac=f.grad, method="BFGS")
    np.testing.assert_allclose(r.x, [3.0, -3.0], atol=1e-6)


def test_compile_fit_exponential_decay():
    a, b, t = fr.symbols("a b t")
    rng = np.random.default_rng(1)
    t_data = np.linspace(0, 5, 5000)
    y_data = 2.5 * np.exp(-1.3 * t_data) + rng.normal(0, 0.01, t_data.size)

    res, jac = fr.compile_fit(a * fr.exp(-b * t), [a, b], t, t_data, y_data)
    fit = least_squares(res, [1.0, 1.0], jac=jac)
    assert fit.success
    np.testing.assert_allclose(fit.x, [2.5, 1.3], atol=0.01)
    # jac 形状 = (m 数据点, n 参数)
    assert jac([1.0, 1.0]).shape == (5000, 2)
