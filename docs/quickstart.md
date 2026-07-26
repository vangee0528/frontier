# 快速上手

## 安装

```bash
pip install frontier-symbolic        # 依赖：numpy、llvmlite（均为纯 pip 安装）
```

从源码构建：

```bash
git clone https://github.com/vangee0528/frontier.git
cd frontier
```

然后在仓库根目录：

```bash
pip install "pybind11<3" llvmlite numpy sympy scipy pytest ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build                     # 扩展模块落到 python/frontier/
python -m pytest tests -q              # 跑测试确认环境（conftest 自动定位源码树）
```

自己的脚本使用源码树构建时，需把 `python/` 加入搜索路径：
`PYTHONPATH=python python your_script.py`（`pip install .` 安装后则不需要）。

要求：Python ≥ 3.10，C++20 编译器（GCC 11+ / Clang / MSVC / MinGW-w64），
CMake ≥ 3.22。

## 五分钟教程

### 1. 写公式

```python
import frontier as fr

x, y = fr.symbols("x y")          # 也支持 fr.symbols("q:3") → q0, q1, q2
f = fr.sin(x * y) + (x + y) ** 2 / 2
```

写法与 SymPy 一致：运算符重载、`fr.sin/cos/exp/...`、精确有理数
（`fr.rational(1, 3)` 不会变成 0.333…）。

### 2. 求导

```python
df_dx = fr.diff(f, x)             # y*cos(x*y) + x + y
grad  = fr.grad(f, [x, y])        # 梯度列表
H     = fr.hessian(f, [x, y])     # Hessian（利用对称性）
```

结果自动化简——`x - x` 就是 `0`，`x/x` 就是 `1`，不需要手动 simplify。

### 3. 编译执行

```python
import numpy as np

g = fr.compile(grad, args=(x, y))         # 符号 → LLVM → 机器码
gx, gy = g(np.random.rand(1_000_000),      # 直接吃 NumPy 数组
           np.random.rand(1_000_000))
```

编译一次（几十毫秒量级），之后每次调用都是机器码级的单遍融合循环
——无中间数组、公共子表达式只算一次、SIMD 向量化。

### 4. 从 SymPy 迁移（如果你已有 SymPy 代码）

```python
f = fr.lambdify(args, exprs)      # 与 sympy.lambdify 同签名，一行替换
```

推导代码零改动。详见[用户指南](guide.md)的迁移一节。

## 下一步

- [用户指南](guide.md) —— 核心概念、编译选项、scipy 集成
- [API 参考](api.md) —— 全部公开接口
- [性能指南](performance.md) —— 什么时候快、为什么快、怎么调
- `examples/` —— 三个完整可运行示例
