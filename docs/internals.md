# 设计与内部实现

> 面向贡献者与好奇的用户。使用层面的内容见[用户指南](guide.md)。

## 总体管线

```
Python API（贴近 SymPy 习惯）
   │  构造（builders：构造期规范化）
   ▼
符号表达式 —— C++ 不可变 DAG，全局 hash-consing
   │  变换：diff / subs（DAG memo 化）
   ▼
Tape —— 后端无关的线性 SSA 指令序列（CSE 在此天然完成）
   │  代码生成（CodegenBackend 接口）
   ▼
LLVM IR 文本 ──llvmlite JIT──▶ 机器码批量核函数
```

一个刻意的架构决策：C++ 侧只产出 LLVM IR **文本**，JIT 由 pip 即装的
llvmlite 完成。这避免了链接 LLVM C++ 库（尤其在 Windows 上），
并让 IR 可打印、可测试（`CompiledFunction.ir`）。

## 分层

```
python/frontier/   API 门面、compile()、llvmlite JIT、SymPy 互操作、scipy 适配器
bindings/          pybind11 薄绑定——只做类型转换与异常映射，零算法逻辑
cpp/               C++20 核心：表示（number/expr/builders/registry）、
                   变换（diff/subs）、降级与代码生成（tape/codegen）
```

依赖单向：变换层依赖核心表示，代码生成依赖两者。绑定层无逻辑，
保证未来开放 C++ 用户 API 无需迁移代码。

## 核心表示

### 精确数值塔

`Number = Int(int64) | Rational(int64/int64) | Real(double)`。
精确类型间的运算保持精确（Knuth 预约分 + checked 溢出检测，
溢出降级 Real）；只有 Real 参与才引入浮点。无 `__int128` 依赖
（`checked_int.hpp` 抽象 GCC/Clang builtin 与 MSVC intrinsic）。

### 表达式 DAG 与 hash-consing

六种结构性节点：`Constant / Symbol / Add / Mul / Pow / Func`。
Add 与 Mul 用"系数 + 有序项表"表示（SymEngine 思路），
同类项/同底幂合并在**构造期**完成——系统里任何可达表达式恒为规范形，
构造入口（builders）是唯一路径，节点构造函数私有。

所有节点经全局 intern 表去重：**指针相等 ⇔ 结构相等**。
这一个机制同时给出 O(1) 相等判断、免费 CSE、求导 memo、
以及"测试可以用指针断言"。

两个从实战中学来的细节：

- 深表达式链（数千层）的析构走迭代队列而非 shared_ptr 递归释放
  （否则栈溢出）；队列对象按线程刻意泄漏，规避 CRT 收尾阶段
  TLS 已析构而 Deleter 仍运行的 UB；
- 常量单例（0/1/-1）同样刻意泄漏，理由相同。

### 函数即数据

数学函数不是节点子类。`sin(x)` 是 `Func{id, args}`，`id` 指向注册表
中的一条 `FuncDef`：

```cpp
struct FuncDef {
    std::string name;      int arity;
    DerivFn     deriv;     // 符号导数规则
    double (*eval)(const double*);   // 常量折叠
    CodegenSpec codegen;   // intrinsic / libm / 自定义降级
    RewriteFn   rewrite;   // 可选规范化（如 sqrt(x) → x^(1/2)）
};
```

新增函数注册一条记录，求导、折叠、代码生成、Python 端包装
（按注册表自动生成）全部自动获得。

## 代码生成

Tape 是符号与数值世界的边界：之上精确算术，之下 f64。
`LlvmTextBackend` 发射批量核函数 `void kernel(double** ins,
double** outs, i64 n)`——对 `[0, n)` 的紧循环。

关键的发射策略：

- **向量化数学**：sin/cos/exp/log（及组合出的 tan/tanh）以 fdlibm
  风格的范围规约 + 多项式实现，作为 `alwaysinline` IR 函数发射进
  模块。内联后循环体是纯算术 + select，LLVM 循环向量化器可整体
  SIMD 化——libm 调用会把循环切成标量段，这是 NumPy 路线的天花板；
- **Pow 特化**：±0.5 → sqrt / rsqrt，-1 → fdiv，小整数 → 乘法链
  （平方法展开，避免 powi 标量调用切割向量化），大整数 → powi；
- **uniform 输入**：在 entry block 载入一次，LLVM LICM 自动把只
  依赖它们的子表达式提升出循环；
- 条件（where/比较）→ `fcmp` + `select`，无分支。

扩展后端（C 源码、CUDA/NVVM）= 实现一个 `CodegenBackend`，
Tape 与上层不变。

## 不变量（贡献时必须守住）

1. 任何可达 `Expr` 恒为规范形（只能经 builders 构造）；
2. intern 表内指针相等 ⇔ 结构相等；
3. 符号层不引入浮点误差（只有 Real 参与才有 double 运算）；
4. 绑定层无算法逻辑；
5. Tape 之下不回流（代码生成不得反查符号层）。

## 测试策略

- **C++（doctest）**：数值塔算术、规范化不变量（含指针相等断言）、
  求导规则、Tape/CSE 指令计数；
- **pytest**：API 行为、与 SymPy 交叉验证（同一表达式两边独立求导
  求值，rtol 1e-9）、线程安全、pickle、退出崩溃回归；
- **差分模糊**（`tests/fuzz/`）：随机表达式三方对拍
  （Frontier / sympy.lambdify / sympy 高精度 evalf 仲裁），
  小批量纳入 CI 冒烟，大批量手动跑。
