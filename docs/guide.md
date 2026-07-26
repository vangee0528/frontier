# 用户指南

## Frontier 是什么

Frontier 是一条**符号 → 数值的编译管线**：用贴近 SymPy 的语法书写
符号公式，C++ 内核完成求导与化简，LLVM JIT 编译为机器码级批量函数。
它服务一个明确场景：**推导一次、求值百万次**——优化、拟合、仿真、
ODE 积分中的模型函数与导数。

它不是完整的 CAS：不做积分、方程求解、三角恒等式化简。
这些工作交给 SymPy 完成，再经 `fr.from_sympy` 无缝接入。

## 核心概念

### 表达式

```python
x, y = fr.symbols("x y")
e = fr.sin(x) * y + fr.rational(1, 3)
```

- **不可变**且**全局去重**：结构相同的表达式是同一个对象，
  `x + y == y + x` 为真，可作 dict key。
- **精确算术**：整数与有理数运算保持精确，`fr.sqrt(fr.as_expr(2))`
  保持符号形式不掉精度；只有浮点常量参与时才引入浮点。
- **恒为规范形**：构造即化简（同类项合并、常量折叠），
  没有"忘了调 simplify"这回事。
- 可 `pickle` / `copy.deepcopy`，跨进程可用。

### 编译函数

```python
g = fr.compile(exprs, args, **options)
```

`exprs` 是单个表达式或列表（多输出）；`args` 是输入符号的顺序。
返回的 `CompiledFunction` 有三种调用方式：

| 调用 | 适用场景 |
| --- | --- |
| `g(*arrays)` | 通用批量：一维 float64 数组（标量自动广播），多输出返回元组 |
| `g.eval_stacked(*arrays)` | 多输出大批量：直接返回 `(n_outputs, n)` 二维数组，零拷贝 |
| `g.eval_scalars(*vals)` | 逐点调用（ODE 右端、优化器回调）：微秒级单点快路径 |

编译选项（细节与调优见[性能指南](performance.md)）：

- `uniform=[...]`——声明"整个批量中取同一标量"的参数（如拟合参数），
  相关子表达式被提升出循环，常带来数倍收益；
- `workers=N | "auto"`——大批量多线程分块；
- `cache=True`——编译产物落盘，跨进程免重编译；
- `vecmath=True`（默认）——超越函数用可向量化实现（~1-2 ulp）；
- `fastmath=False`（默认）——浮点松弛优化，显式开启。

### 异常

`FrontierError`（基类）⊃ `DomainError`（数学域错误，如 1/0）、
`CompileError`（编译期错误，如自由符号缺失——报错会列出全部缺失
符号与当前参数表）。

## 从 SymPy 迁移

```python
import frontier as fr
f = fr.lambdify(args, exprs)          # 原地替换 sympy.lambdify
```

支持：单表达式 / 列表 / `sympy.Matrix`（输出保持矩阵形状）、
单符号参数形式、`modules=`/`cse=` 参数（接受并忽略）。
表达式覆盖：初等函数全集、`Piecewise`、`Heaviside`、关系算子与
`And/Or/Not`、n 元 `Max/Min`、`erf/erfc`、`floor/ceiling`、`atan2`。

与 `sympy.lambdify` 的行为差异（完整清单见性能指南末节）：

- 输入限标量 / 一维 float64 数组；二维网格请 `X.ravel()` 后 reshape 输出；
- 复数输入直接报 `TypeError`（不静默取实部）；
- 首次调用前有一次 JIT 编译开销；
- 不支持的 sympy 节点明确抛 `NotImplementedError` 并列出可用函数集。

已有表达式也可以显式转换后混用 Frontier API：

```python
e = fr.from_sympy(sympy_expr)
fr.diff(e, fr.symbol("x"))
```

## SciPy 集成

三个适配器把"模型 + 自动导数"打包成 scipy 的调用约定：

```python
# ODE：solve_ivp(rhs, ..., jac=jac)
ode = fr.compile_ode(rhs_exprs, state=[x, v], params=[k])
solve_ivp(ode.rhs, t_span, y0, jac=ode.jac, args=(2.0,), method="BDF")

# 优化：minimize(f, x0, jac=f.grad, hess=f.hess)
obj = fr.compile_objective(expr, [x, y])
minimize(obj, x0, jac=obj.grad, hess=obj.hess, method="trust-ncg")

# 拟合：least_squares(res, p0, jac=jac)
res, jac = fr.compile_fit(model, params, x, x_data, y_data)
least_squares(res, p0, jac=jac)
```

注意 `compile_ode` 返回的对象内部复用缓冲，**实例不跨线程共享**
（并行积分时每线程各建一个，重复构建有进程内缓存兜底）。

## 线程与进程

- `Expr` 不可变，任意共享；
- `CompiledFunction` 的三种调用路径均线程安全；
- `Expr` 与 `CompiledFunction` 均可 pickle——joblib、
  `multiprocessing`、`concurrent.futures` 子进程可直接传递
  （子进程重新 JIT；配合 `cache=True` 几乎零开销）。

## 扩展函数集

数学函数是注册表数据而非硬编码：在 C++ 侧注册一条 `FuncDef`
（名字、导数规则、数值求值、代码生成方式），求导、常量折叠、
编译、Python 端包装全部自动获得。参见 [internals.md](internals.md)。
