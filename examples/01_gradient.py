"""符号梯度 → 编译内核：Frontier 的最小工作流。

运行：python examples/01_gradient.py
"""
import numpy as np

import frontier as fr

# 1. 用符号写公式（写法与 SymPy 一致）
x, y, z = fr.symbols("x y z")
f = fr.sin(x * y) + fr.exp(-z**2) * (x + y) ** 2

# 2. 符号求梯度，编译为机器码批量函数
grad = fr.grad(f, [x, y, z])
g = fr.compile(grad, args=(x, y, z))

# 3. 在一百万个点上批量求值
rng = np.random.default_rng(0)
xs, ys, zs = (rng.random(1_000_000) for _ in range(3))
gx, gy, gz = g(xs, ys, zs)

print("grad f =", [str(e) for e in grad])
print("gx[:3] =", gx[:3])
