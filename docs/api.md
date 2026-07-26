# API 参考

模块导入约定：`import frontier as fr`。

## 符号与表达式构造

| 接口 | 说明 |
| --- | --- |
| `fr.symbols(names)` | 空格/逗号分隔批量创建；支持 range 语法 `"q:3"`（q0 q1 q2）、`"a:d"`（a b c d）；单名字直接返回符号 |
| `fr.symbol(name)` | 创建单个符号 |
| `fr.as_expr(v)` | int / float / Expr → Expr（int 走精确整数） |
| `fr.rational(p, q)` | 精确有理数 p/q |
| `fr.pi`, `fr.E` | 常量（f64 精度的 Real） |
| 运算符 | `+ - * / ** -`（负号）；比较 `< <= > >=` 产生 0/1 指示表达式 |

**数学函数**（模块级，接受 Expr 或数值）：
`sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh
exp log sqrt abs sign atan2 max min erf erfc floor ceil`

| 接口 | 说明 |
| --- | --- |
| `fr.where(cond, a, b)` | 条件选择（编译为无分支 select）；导数按分支传播 |

## Expr 对象

| 成员 | 说明 |
| --- | --- |
| `e.subs({k: v})` | 子表达式替换；键可为任意表达式（如 `{fr.sin(x*y): z}`），结果自动规范化 |
| `e.diff(var)` | 求偏导（等价 `fr.diff(e, var)`） |
| `e.free_symbols` | 自由符号列表（首次出现顺序） |
| `e.is_zero / is_one / is_constant` | 结构判定 |
| `float(e)` | 常量表达式转 float；非常量抛 TypeError |
| `==` / `hash()` | 结构相等（O(1)，全局去重保证）；可作 dict key |
| pickle / deepcopy | 支持 |

## 微分

| 接口 | 说明 |
| --- | --- |
| `fr.diff(e, v, *more)` | 偏导；多变量依序连续求导 |
| `fr.grad(e, vars)` | 梯度列表 |
| `fr.jacobian(exprs, vars)` | 嵌套列表 J[i][j] = ∂exprs[i]/∂vars[j] |
| `fr.hessian(e, vars)` | 二阶导矩阵（对称性优化） |

## 编译

```python
fr.compile(exprs, args, *, fastmath=False, vecmath=True,
           workers=1, cache=False, uniform=None) -> CompiledFunction
```

| 参数 | 说明 |
| --- | --- |
| `exprs` | 单表达式或列表（多输出） |
| `args` | 输入符号序列；调用时按此顺序传值 |
| `uniform` | 批量中取同一标量的参数序列；写错符号会报错 |
| `vecmath` | 超越函数用可向量化实现（~1-2 ulp）；False 严格 libm |
| `workers` | 批量分块线程数；`"auto"` = CPU 核数；批量 < 8192 自动单线程 |
| `cache` | 编译产物落盘（`$FRONTIER_CACHE_DIR` 或 `~/.cache/frontier`） |
| `fastmath` | 浮点重结合等松弛优化 |

**CompiledFunction**：

| 成员 | 说明 |
| --- | --- |
| `f(*arrays)` | 批量求值；标量广播；多输出返回元组（单块缓冲的行视图） |
| `f.eval_stacked(*arrays)` | 返回 `(n_outputs, n)` 二维数组，零拷贝 |
| `f.eval_scalars(*vals)` | 单点快路径 → `(n_outputs,)` 数组；线程安全 |
| `f.ir` / `f.optimized_ir` | 编译前 / LLVM O3 后的 IR 文本（调试用） |
| `f.args` / `f.n_inputs` / `f.n_outputs` | 元信息 |
| pickle | 支持（重载时重新 JIT） |

## SymPy 互操作

| 接口 | 说明 |
| --- | --- |
| `fr.from_sympy(e)` | SymPy 表达式 → Frontier 表达式；不支持的节点抛 NotImplementedError（附可用函数集） |
| `fr.lambdify(args, exprs, **opts)` | `sympy.lambdify` 同签名替代；支持 Matrix / 单符号 / `modules`/`cse`（忽略）；额外接受 compile 的全部选项 |

## SciPy 适配器

| 接口 | 返回 | 用法 |
| --- | --- | --- |
| `fr.compile_ode(rhs, state, t=None, params=())` | `.rhs(t,y,*p)` / `.jac(t,y,*p)` | `solve_ivp(f.rhs, ..., jac=f.jac)`；实例不跨线程 |
| `fr.compile_objective(e, vars, params=())` | 可调用 + `.grad` / `.hess` | `minimize(f, x0, jac=f.grad, hess=f.hess)` |
| `fr.compile_fit(model, params, x, x_data, y_data)` | `(residual, jac)` | `least_squares(res, p0, jac=jac)`；params 自动 uniform |

## 异常

```
FrontierError            # 基类
├── DomainError          # 数学域错误（除零、非法有理数）
└── CompileError         # 编译期错误（自由符号缺失等）
```
