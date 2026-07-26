"""scipy 生态一行接入：ODE 积分与曲线拟合。

运行：python examples/03_ode_fitting.py（需要 scipy）
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

import frontier as fr

# ---------- ODE：谐振子，rhs + 解析 Jacobian 一次生成 ----------
x, v, k = fr.symbols("x v k")
ode = fr.compile_ode([v, -k * x], state=[x, v], params=[k])
sol = solve_ivp(ode.rhs, (0, 2 * np.pi), [1.0, 0.0],
                jac=ode.jac, args=(1.0,), method="Radau", rtol=1e-9)
print("harmonic oscillator, back to start:", sol.y[:, -1])

# ---------- 拟合：指数衰减，残差 + 解析 Jacobian 一次生成 ----------
a, b, t = fr.symbols("a b t")
rng = np.random.default_rng(1)
t_data = np.linspace(0, 5, 10_000)
y_data = 2.5 * np.exp(-1.3 * t_data) + rng.normal(0, 0.01, t_data.size)

residual, jac = fr.compile_fit(a * fr.exp(-b * t), [a, b], t, t_data, y_data)
fit = least_squares(residual, [1.0, 1.0], jac=jac)
print("fitted (a, b) =", fit.x, " true = [2.5, 1.3]")
