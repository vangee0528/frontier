#pragma once

#include "frontier/codegen/backend.hpp"

namespace frontier {

// 发射 LLVM IR 文本模块（typed pointer 语法，兼容 llvmlite 0.4x 的 LLVM）。
// 内核签名：void kernel(double** ins, double** outs, i64 n)
//   ins / outs 均为「指向 f64 数组的指针」的数组；对 [0,n) 做紧循环，
//   向量化交由 LLVM O3（loop/SLP vectorizer）完成。
//
// Pow 特化：整数指数 → llvm.powi；±0.5 → sqrt / 1/sqrt；-1 → fdiv；
// 其余 → llvm.pow。函数调用按 FuncRegistry 的 CodegenSpec 发射。
class LlvmTextBackend final : public CodegenBackend {
public:
    std::string name() const override { return "llvm-ir-text"; }
    std::string emit(const Tape& tape, const KernelSpec& spec) const override;
};

}  // namespace frontier
