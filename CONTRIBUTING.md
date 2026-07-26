# 贡献指南

## 开发环境

```bash
pip install pybind11 llvmlite numpy sympy scipy pytest ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build                 # 产出 python/frontier/_core.*
ctest --test-dir build              # C++ 单元测试（doctest）
PYTHONPATH=python python -m pytest tests -q
```

调试构建（启用规范形断言）：`-DCMAKE_BUILD_TYPE=Debug`。
提交前请在 Debug 下至少跑一遍 C++ 测试——多数表示层不变量只在
断言中检查。

## 提交要求

- 新功能必须带测试；修 bug 必须带能复现该 bug 的回归测试；
- C++ 改动同时跑 `ctest` 与 pytest（绑定层行为经 Python 测试覆盖）；
- 涉及数值行为的改动，跑一轮差分模糊冒烟：
  `PYTHONPATH=python python tests/fuzz/fuzz_differential.py --seed 1 --n 100`
  （发现 BUG 类记录时退出码非零；类别定义见该文件头部）
- 涉及表示层（expr/builders/number）的改动，先读
  [docs/internals.md](docs/internals.md) 的不变量一节——
  它们是这个库正确性的地基。

## 常见贡献路径

**新增数学函数**（最常见）：在 `cpp/src/func_registry.cpp` 注册一条
`FuncDef`（名字、元数、导数规则、数值求值、codegen 方式），
求导/常量折叠/编译/Python 包装自动获得。参考 `erf` 的注册即可。
请同时在 `tests/` 加数值与导数的对拍用例。

**新增化简 pass**：实现 `ExprPass` 接口（`cpp/include/frontier/pass.hpp`）。

**新增代码生成后端**：实现 `CodegenBackend` 接口
（`cpp/include/frontier/codegen/backend.hpp`），输入 Tape 输出目标代码
文本；不要反向依赖符号层。

## 代码风格

- C++：4 空格缩进，遵循现有命名（类型 PascalCase、函数/变量
  snake_case）；注释解释"为什么"而非"是什么"；
- Python：面向公共 API 写 docstring（现有风格：中文、
  numpy-doc 简化版）；
- 绑定层（bindings/）禁止算法逻辑——只做类型转换与异常映射。

## 报告问题

请附：最小复现代码、`frontier.__version__`、平台/编译器、
若为数值问题请附 `CompiledFunction.ir` 输出。
