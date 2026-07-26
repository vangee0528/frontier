# 案例集：在真实生态中使用 Frontier

五个来自真实开源生态的改造模式。每节给出场景、改造代码与实测结果
（作者环境：Windows 11 / 16 核 / NumPy 2.2；相对比例跨机器稳定），
并如实标注**没有优势的场景**。基准套件的可复现入口见
`benchmarks/run_all.py`。

---

## 1. 多体动力学：sympy.physics.mechanics / PyDy

**场景**：`KanesMethod` 符号推导 N 连杆摆的运动方程
`M(x)·ẋ = F(x)`，数值化后交给 ODE 积分器。这是机器人学/生物力学
的标准工作流，教科书写法用 `sympy.lambdify`。

**改造**（推导代码零改动，只换数值化一行）：

```python
# 原： f = sympy.lambdify(args, entries, modules="numpy", cse=True)
f = fr.lambdify(args, entries, workers="auto")
```

逐步积分场景用单点快路径：

```python
def rhs(x):
    flat = f.eval_scalars(*x, *p_vals)      # 一次 C 调用出全部 M、F 元素
    return np.linalg.solve(flat[:n*n].reshape(n, n), flat[n*n:])
```

**结果**：ODE 单轨迹积分 2.8×（5 连杆）～ 5.5×（8 连杆）——
lambdify 的 Python 解释开销随表达式数线性增长，编译内核不变；
批量 ensemble 求值 3.1×。

---

## 2. 光谱拟合：lmfit

**场景**：多峰光谱拟合。lmfit 的 Levenberg-Marquardt 默认用
数值差分近似 Jacobian：每步迭代额外做 n_params 次全量残差求值。

**改造**：模型公式符号化写一遍，解析 Jacobian 由 `fr.grad` 自动生成，
经 lmfit 公开接口注入；`uniform` 声明拟合参数：

```python
f_jac = fr.compile(fr.grad(model, params), args=(*params, x), uniform=params)

def jac(p):
    return f_jac.eval_stacked(*[p[k].value for k in names], x_data)

lmfit.Minimizer(residual, params).leastsq(Dfun=jac, col_deriv=1)
```

**结果**（11 参数 3 高斯峰 @ 10 万点）：拟合 2.5×，残差求值次数
98 → 10；`uniform` 使高斯归一化因子被提升出批量循环，单次模型求值
反超手写 NumPy 约 2×。解析导数还消除了差分步长噪声——病态问题上
这常是"收敛与否"的差别。

---

## 3. 刚性化学动力学：chempy / pyodesys

**场景**：反应网络 → 质量作用速率表达式 → 刚性积分器（BDF/Radau）。
隐式步进每步都要 RHS + Jacobian。

**改造**：

```python
ode = fr.compile_ode(rhs_exprs, state=state)          # 一行
solve_ivp(ode.rhs, t_span, y0, jac=ode.jac, method="BDF")
```

**结果**：单步成本恒定 ~2µs，不随体系规模变化；40 组分网络的
40×40 Jacobian 对 lambdify（已用对其最有利的 Matrix+cse 写法）
达 **36×**，且随规模继续增长。

**诚实记录**：端到端积分只有 1.3–1.5×——scipy 求解器自身的
Python 开销（LU、步长控制）占大头。小体系（≤5 组分）单点 RHS
与 lambdify 打平：跨 C 边界的 ~2µs 底噪省不掉。

---

## 4. 符号拟合框架：symfit

**场景**：symfit 的数值化核心 `sympy_to_py` 是 lambdify 薄封装，
模型与自动求导的 Jacobian 都经它数值化。

**改造**：monkeypatch 一处，用户代码零改动；关键技巧是把框架的
`Parameter` 类型自动映射为 `uniform` 声明：

```python
def frontier_sympy_to_py(func, args):
    uniform = [a for a in args if isinstance(a, symfit.Parameter)] or None
    compiled = fr.lambdify(list(args), func, uniform=uniform)
    ...   # 失败自动回退原实现，对任意模型都是安全的渐进增强

symfit.core.support.sympy_to_py = frontier_sympy_to_py
```

**结果**（50 万点双高斯+指数背景，7 参数）：端到端拟合 2.3×。
框架里现成的类型信息（参数 vs 数据变量）直接换来编译器优化，
是"类型信息换性能"的典型样例。

---

## 5. 天文模型拟合：astropy.modeling

**场景**：astropy 的单个内置模型带手写 `fit_deriv` 解析导数，但
**复合模型**（多峰+连续谱）没有——`LevMarLSQFitter` 回退有限差分。
手写导数无法组合，符号求导天生可组合。

**改造**：

```python
model = fgauss(a1,m1,s1) + fgauss(a2,m2,s2) + fgauss(a3,m3,s3) + c0 + c1*(x-x0)
res_fn, jac_fn = fr.compile_fit(model, params, x, x_data, y_data)
fit = least_squares(res_fn, p0, jac=jac_fn, method="lm")
```

**结果**（20 万点、三发射线+线性连续谱、11 参数）：5.43s → 0.82s
（**6.6×**），参数误差同级。加速三层叠加：解析导数省掉每步 11 次
差分求值、`uniform` 提升归一化因子、编译内核本身快于逐模型 NumPy。

---

## 模式总结

| 模式 | 适用 | 入口 |
| --- | --- | --- |
| lambdify 一行替换 | 已有 SymPy 推导代码 | `fr.lambdify` |
| 解析导数注入 | 任何用差分 Jacobian 的拟合器 | `fr.grad` + `compile(uniform=...)` |
| scipy 适配器 | solve_ivp / minimize / least_squares | `compile_ode/objective/fit` |
| 框架内核替换 | 数值化入口集中的框架 | monkeypatch + 类型→uniform 映射 |

选型经验：表达式越大、批量越大、导数越贵，收益越大；
≤5 变量的单点求值别指望提速。
