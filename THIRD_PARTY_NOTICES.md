# 第三方代码声明

本项目包含或衍生自以下第三方代码：

## doctest（测试框架，vendored）

- 文件：`cpp/tests/vendor/doctest.h`
- 来源：https://github.com/doctest/doctest （v2.4.11）
- 许可：MIT License, Copyright (c) 2016-2023 Viktor Kirilov
- 用途：C++ 单元测试，仅参与测试构建，不进入发布二进制

## fdlibm 派生的数学函数系数

- 文件：`cpp/src/codegen_vecmath.cpp`
- `sin`/`cos` 的多项式系数与 Cody-Waite 范围规约常数、`log` 的
  atanh 级数系数派生自 fdlibm（Freely Distributable LIBM）：

  > Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
  >
  > Developed at SunSoft, a Sun Microsystems, Inc. business.
  > Permission to use, copy, modify, and distribute this software is
  > freely granted, provided that this notice is preserved.

- `exp` 的泰勒系数（1/k!）与 `tanh` 的泰勒系数为数学常数，
  实现结构为独立编写；`erf`/其余函数直接调用平台 libm，无派生代码。

## 其余构建期依赖（不随发布二进制分发）

- pybind11（BSD-3-Clause）——绑定层头文件库
- llvmlite（BSD-2-Clause）——Python 侧 JIT 运行时依赖（pip 安装）
