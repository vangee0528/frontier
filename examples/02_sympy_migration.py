"""从 SymPy 迁移：lambdify 一行替换。

已有的 SymPy 推导代码不需要任何改动，
只把最后的 sympy.lambdify 换成 fr.lambdify。

运行：python examples/02_sympy_migration.py（需要 sympy）
"""
import numpy as np
import sympy as sp

import frontier as fr

# ---- 这一段是"你已有的 SymPy 代码"，原样保留 ----
x, a, b = sp.symbols("x a b")
model = a * sp.exp(-b * x**2) * sp.cos(x)
dmodel_da = sp.diff(model, a)
dmodel_db = sp.diff(model, b)

# ---- 迁移点：sympy.lambdify -> fr.lambdify（同签名）----
f = fr.lambdify([x, a, b], [model, dmodel_da, dmodel_db],
                uniform=[a, b])   # 参数声明 uniform：只依赖参数的
                                  # 子表达式被提升出批量循环

xs = np.linspace(-3, 3, 100_000)
val, d_a, d_b = f(xs, 2.0, 0.5)
print("value[:3] =", val[:3])

# 对照 sympy.lambdify 验证一致
ref = sp.lambdify([x, a, b], model, modules="numpy")(xs, 2.0, 0.5)
assert np.allclose(val, ref)
print("matches sympy.lambdify: True")
