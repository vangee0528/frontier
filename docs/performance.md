# 性能指南

## 为什么快

`fr.compile` 产出的是**单遍融合循环**的机器码：

1. **无中间数组**——NumPy 表达式每个中间结果都要写一遍内存再读回来，
   编译内核在寄存器里算完整个表达式；
2. **CSE 免费**——共享子表达式在表达式构造期就被全局去重，
   降级时天然只发射一条指令；
3. **SIMD**——sin/cos/exp/log/tan/tanh 采用可内联的多项式实现
   （实测最大误差 ~1-2 ulp），使 LLVM 能把整个循环向量化，
   这是调用 libm 的方案（包括 NumPy）做不到的；
4. **参数外提**——`uniform` 声明的参数在循环外载入，只依赖参数的
   子表达式（归一化因子等）被 LLVM 整体提升出循环。

## 基准结果摘要

测量环境：Windows 11 / MinGW-w64 GCC 14 / 16 核 / NumPy 2.2 /
llvmlite 0.44。不同机器绝对值会变，相对关系稳定。
第一行的梯度基准可直接在仓库内复现（`python benchmarks/bench_gradient.py`）；
生态改造行的数字为作者环境对各库公开工作流的实测，
测量脚本未随库分发，表中注明了对照方的写法以便独立复现：

| 场景 | 对比对象 | 结果 |
| --- | --- | ---: |
| 3 变量梯度 @ 100 万点 | sympy lambdify | **~13×** |
| 同上 | 手写 NumPy（人肉求导+向量化） | **~7×** |
| 多体动力学 ODE 单轨迹（sympy.physics.mechanics，10~16 维） | lambdify | 2.8×~5.5× |
| 刚性动力学单步 Jacobian（40 组分反应网络） | lambdify（Matrix+cse 最优写法） | **36×**，且随规模增长 |
| lmfit 11 参数光谱拟合 @ 10 万点 | lmfit 数值差分 | 2.5×（残差求值 98→10 次） |
| astropy 复合模型拟合 @ 20 万点 | LevMar 有限差分 | 6.6×（同精度） |

**没赢的地方（同样重要）**：

- 小体系（≤5 变量）的单点求值与 lambdify 打平——跨 C 边界的
  ~2µs 底噪省不掉；
- 端到端 ODE 积分加速比远小于单步加速比——scipy 求解器自身的
  Python 开销（LU、步长控制）占大头，任何 RHS 优化都被摊薄。

## 调用路径怎么选

| 路径 | 场景 |
| --- | --- |
| `f(*arrays)` | 通用批量 |
| `f.eval_stacked(*arrays)` | 多输出大批量（Jacobian、质量矩阵）——省一次 stack 拷贝 |
| `f.eval_scalars(*vals)` | 每次一个点（ODE 右端、优化器回调） |

## 编译开关怎么选

| 开关 | 默认 | 何时改 |
| --- | --- | --- |
| `uniform=[...]` | 无 | **有拟合参数/物理常数就声明**——常是最大单项收益（光谱案例 4.5ms→0.9ms） |
| `vecmath` | True | 需要与 libm 逐位一致时关（吞吐掉数倍）。已知域限制：sin/cos 在 \|x\| ≳ 1e8 精度渐降；log 不支持次正规输入 |
| `workers` | 1 | 单次批量 ≥ 10 万点时开 `"auto"`；小批量高频调用别开 |
| `cache` | False | CLI 工具、批处理、multiprocessing 场景开 |
| `fastmath` | False | 确认有收益再开（vecmath 已解决超越函数瓶颈，此开关通常收益很小且丢位一致性） |

## 编译本身的开销

符号求导 + CSE + LLVM O3 + JIT ≈ 数十至数百毫秒（随表达式规模）。
设计前提是一次编译、海量调用；不要在循环里反复 `fr.compile`
（进程内有按 IR 的 LRU 缓存兜底）。

## 与 sympy.lambdify 的全部已知差异

- 输入限标量 / 一维 float64 数组；网格 `ravel()` + reshape；
- 复数输入显式报错（lambdify 静默降级）；
- 标量输入返回长度 1 数组而非 Python float；
- 首次调用前有 JIT 编译开销；
- `modules=` / `cse=` 接受但忽略；
- 数值域为 f64 实数：负底数的非整数次幂返回 NaN
  （sympy 走复数链路可能给出实数结果）；
- `vecmath` 默认开启（~1-2 ulp vs libm；`vecmath=False` 完全一致）。

## 正确性保障

- 与 SymPy 的交叉验证测试常驻（值 + 梯度，rtol 1e-9）；
- 差分模糊测试：随机表达式三方对拍（Frontier vs lambdify vs
  sympy 30 位高精度仲裁），发布前累计约 4000 例，
  详见 `tests/fuzz/`（已纳入 CI 冒烟）。
