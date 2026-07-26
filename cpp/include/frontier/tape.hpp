#pragma once

#include <span>
#include <vector>

#include "frontier/expr.hpp"

namespace frontier {

// Tape：后端无关的线性 SSA 指令序列，符号世界与数值世界的边界。
// 它之上是精确算术，之下是 f64 浮点（未来 f32/复数经类型标注扩展）。
//
// hash-consing 使 CSE 在降级时天然完成：DAG 中共享的节点只发射一条指令。
struct TapeOp {
    enum class Kind : uint8_t {
        Input,  // 读取第 input_index 个参数
        Const,  // f64 常量
        Add,    // v[a] + v[b]
        Mul,    // v[a] * v[b]
        Pow,    // v[a] ^ v[b]（后端可对常量指数特化：整数幂/开方/倒数）
        Call,   // 注册表函数调用
    };

    Kind kind;
    int32_t a = -1, b = -1;         // Add/Mul/Pow 的操作数值编号
    double cval = 0.0;              // Const
    int32_t input_index = -1;       // Input
    FuncId func = 0;                // Call
    std::vector<int32_t> args;      // Call 实参值编号
};

struct Tape {
    std::vector<TapeOp> ops;        // SSA：第 i 条指令的结果编号即 i
    std::vector<int32_t> outputs;   // 每个输出表达式对应的值编号
    size_t num_inputs = 0;
};

// 把一组表达式降级为 Tape。args 必须全为 Symbol；
// 表达式中出现 args 之外的自由符号时抛 CompileError。
Tape lower(std::span<const ExprPtr> exprs, std::span<const ExprPtr> args);

}  // namespace frontier
