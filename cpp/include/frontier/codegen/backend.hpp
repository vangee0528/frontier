#pragma once

#include <string>
#include <vector>

#include "frontier/tape.hpp"

namespace frontier {

// 批量核函数的生成参数
struct KernelSpec {
    std::string name = "frontier_kernel";
    // fast-math：允许重结合/倒数近似等浮点松弛（用户显式开启）
    bool fastmath = false;
    // 向量化数学：sin/cos/exp/log 用可内联多项式实现替代 libm 调用，
    // 解锁循环 SIMD 向量化（误差 1-2 ulp，见 codegen/vecmath.hpp）
    bool vecmath = true;
    // uniform 输入掩码（空 = 全批量）：标记为 uniform 的输入在整个批量
    // 中取同一值（如拟合参数），在 entry block 载入一次（ins[i][0]），
    // LLVM LICM 随之把只依赖 uniform 的子表达式提升出循环。
    std::vector<bool> uniform_inputs;
};

// 代码生成后端接口。输入 Tape（后端无关 SSA），输出目标代码文本。
// v1 唯一实现：LlvmTextBackend；未来 C 源码 / CUDA 后端在此扩展，
// Tape 与其上各层不变（ARCHITECTURE.md §5.2）。
class CodegenBackend {
public:
    virtual ~CodegenBackend() = default;
    virtual std::string name() const = 0;
    virtual std::string emit(const Tape& tape, const KernelSpec& spec) const = 0;
};

}  // namespace frontier
