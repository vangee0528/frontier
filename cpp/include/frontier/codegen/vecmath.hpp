#pragma once

#include <vector>

namespace frontier::vecmath {

// 注册表函数名 → 可向量化 IR 辅助函数名（"fr_sin" 等）；无实现返回 nullptr。
//
// 这些辅助函数是 fdlibm 风格的范围规约 + 多项式实现，以 alwaysinline
// internal 函数发射进模块：内联后循环体变为纯算术 + select + 位操作，
// LLVM 循环向量化器可将其整体 SIMD 化（libm 标量调用做不到这一点）。
//
// 精度：典型误差 1-2 ulp。已知域限制（文档承诺）：
//  - sin/cos：|x| ≲ 1e8 内精度完整（两段 Cody-Waite 规约），更大参数渐降；
//  - log：不处理次正规数输入；log(0) = -inf，log(负) = NaN；
//  - exp：溢出/下溢正确钳制为 inf/0。
const char* helper_for(const char* registry_name);

// 辅助函数的 IR 定义文本
const char* helper_ir(const char* helper_name);

// 该辅助函数调用的其他辅助函数（发射时需一并发射）
std::vector<const char*> helper_deps(const char* helper_name);

}  // namespace frontier::vecmath
