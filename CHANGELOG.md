# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 约定；
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2026-07-26

首个公开版本。

### 新增

**符号核心（C++20）**
- 不可变表达式 DAG，全局 hash-consing（指针相等 ⇔ 结构相等）
- 精确数值塔 Int / Rational / Real，checked 溢出算术（可移植，
  GCC/Clang builtin 与 MSVC intrinsic）
- 构造期规范化：同类项/同底幂合并、常量折叠、分配律、精确性规则
  （`sqrt(2)`、`sin(0)` 保持符号形式）
- 变换：`diff` / `grad` / `jacobian` / `hessian`（DAG memo）、
  `subs`（任意子表达式替换）
- 注册表驱动的函数集：sin cos tan asin acos atan sinh cosh tanh
  asinh acosh atanh exp log sqrt abs sign atan2 max min erf erfc
  floor ceil，条件原语 `where` 与比较谓词

**编译执行**
- Tape SSA 降级（CSE 随 hash-consing 天然完成）→ LLVM IR 文本 →
  llvmlite O3 JIT 批量核函数
- 向量化超越函数（fdlibm 风格内联 IR，~1-2 ulp，解锁 SIMD 循环）；
  可用 `vecmath=False` 回退 libm
- `uniform` 参数声明（参数子表达式提升出批量循环）
- `workers` 多线程分块、`cache=True` 编译产物磁盘缓存
- 三条调用路径：批量 `__call__`、二维零拷贝 `eval_stacked`、
  单点快路径 `eval_scalars`

**生态集成**
- `fr.from_sympy` / `fr.lambdify`（`sympy.lambdify` 同签名替代）：
  Piecewise、Heaviside、关系算子、And/Or/Not、n 元 Max/Min、
  Matrix 输出、单符号形式
- SciPy 适配器：`compile_ode`（solve_ivp）、`compile_objective`
  （minimize）、`compile_fit`（least_squares）

**工程**
- `Expr` 与 `CompiledFunction` 可 pickle / deepcopy
  （multiprocessing / joblib 可用）
- 线程安全的求值路径；错误信息含参数定位与修复提示
- 差分模糊测试基建（`tests/fuzz/`，三方对拍：Frontier vs
  sympy.lambdify vs sympy 高精度 evalf 仲裁），小批量纳入测试套件

### 平台

- Windows（MinGW-w64 与 MSVC 均实测；MSVC 构建启用
  `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` 以兼容旧版 msvcp140.dll）
- Linux（GCC 11，C++ 核心与全部断言实测）
- macOS（CI 配置就绪，未在本地实测）

### 发布前验证

- 差分模糊测试：两条独立验证线累计超过 4000 条随机表达式
  （值 + 全部一阶导，rtol 1e-9，30 位高精度仲裁），全部缺陷
  在发布前修复并复验，零遗留
- 测试套件：Python 70 例 + C++ 102 断言
- 定向探针：atan2 四象限、`x^x`、负整数幂、`0^0`、深表达式
  （1500 层）构造与析构、进程退出路径、多线程求值

### 已知限制

- 数值域为 f64 实数（复数 / f32 在路线图）
- `vecmath` 的 sin/cos 在 |x| ≳ 1e8 精度渐降；log 不支持次正规输入
- 输入限标量 / 一维数组（网格需 ravel + reshape）
- 端到端 ODE 积分受 scipy 求解器自身开销钳制
  （编译端积分器在路线图）
