# Frontier

[English](README.md) | 简体中文

**符号数学的编译执行层**：用贴近 SymPy 的语法书写公式，编译为
机器码级的批量数值函数。在"推导一次、求值百万次"的场景下
（拟合、优化、仿真、ODE），作为 `sympy.lambdify` 的直接替代。

```python
import frontier as fr
import numpy as np

x, y, z = fr.symbols("x y z")
f = fr.sin(x * y) + fr.exp(-z**2) * (x + y) ** 2

g = fr.compile(fr.grad(f, [x, y, z]), args=(x, y, z))   # 符号梯度 → 机器码

xs, ys, zs = (np.random.rand(1_000_000) for _ in range(3))
gx, gy, gz = g(xs, ys, zs)
```

已有 SymPy 项目一行迁移：

```python
f = fr.lambdify(args, exprs)     # 与 sympy.lambdify 同签名
```

## 特性

- **快**：单遍融合循环 + CSE + SIMD 向量化超越函数 + 参数外提。
  典型批量场景比 `sympy.lambdify` 快 3–36×，多数场景快于手写 NumPy
  （数字与复现方式见[性能指南](docs/performance.md)）
- **自动导数**：`diff` / `grad` / `jacobian` / `hessian`，
  解析精度，无差分噪声；`compile_ode` / `compile_objective` /
  `compile_fit` 一行接入 scipy
- **符号层保证正确**：精确整数/有理数算术、构造即自动化简、
  表达式全局去重（结构相等判断 O(1)）
- **工程可用**：线程安全、可 pickle（multiprocessing/joblib）、
  编译产物磁盘缓存、错误信息可自救
- **轻依赖**：`numpy` + `llvmlite`，均为纯 pip 安装

## 安装

```bash
pip install frontier-symbolic
```

从源码构建见[快速上手](docs/quickstart.md)。
要求 Python ≥ 3.10；支持 Linux / macOS / Windows。

## 文档

| | |
| --- | --- |
| [快速上手](docs/quickstart.md) | 安装与五分钟教程 |
| [用户指南](docs/guide.md) | 核心概念、编译选项、SymPy 迁移、scipy 集成 |
| [API 参考](docs/api.md) | 全部公开接口 |
| [性能指南](docs/performance.md) | 基准数字、调优、与 lambdify 的差异清单 |
| [案例集](docs/case-studies.md) | 五个真实生态的改造模式与实测 |
| [设计与内部](docs/internals.md) | 架构、不变量、扩展点（贡献者向） |

可运行示例见 [examples/](examples/)。

## 边界（诚实声明）

Frontier 不是完整的 CAS：不做积分、方程求解、三角恒等式化简——
这些交给 SymPy，推导结果经 `fr.from_sympy` 无缝接入。
数值域为 f64 实数；小表达式（≤5 变量）的单点求值与 lambdify 打平，
优势随表达式规模与批量大小增长。

## 正确性

- 与 SymPy 的交叉验证测试（值 + 梯度，rtol 1e-9）随测试套件运行，
  并包含小批量差分模糊冒烟（`tests/fuzz/`）；
- 发布前差分模糊测试累计超过 4000 条随机表达式三方对拍
  （vs `sympy.lambdify` vs sympy 30 位高精度仲裁），零未解决缺陷；
- C++ 核心 102 断言 + Python 测试 70 例；
- 基准套件一键复现：`python benchmarks/run_all.py --quick`
  （批量/逐点/拟合/规模扫描四项，每项内置正确性对拍断言）。

## 许可

MIT，见 [LICENSE](LICENSE)。变更历史见 [CHANGELOG](CHANGELOG.md)。
