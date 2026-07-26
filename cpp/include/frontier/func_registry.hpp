#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "frontier/expr.hpp"

namespace frontier {

// 函数如何降级为数值代码。后端据此发射调用（见 codegen/backend.hpp）。
struct CodegenSpec {
    enum class Lower : uint8_t {
        Intrinsic,  // LLVM intrinsic，symbol 如 "llvm.sin.f64"
        LibmCall,   // 进程内 libm 符号调用，symbol 如 "tan"
        Custom,     // 后端特判发射（如 "sign" 展开为比较+select）
    };
    Lower lower;
    std::string symbol;
};

// 「函数即数据」：新增数学函数 = 注册一条 FuncDef，
// 求导、常量折叠、代码生成、Python 包装全部自动获得。
struct FuncDef {
    std::string name;
    int arity;
    // 对第 arg_index 个参数的偏导表达式（不含链式法则的外层因子）
    std::function<ExprPtr(const std::vector<ExprPtr>& args, int arg_index)> deriv;
    // 数值求值（常量折叠用）；args 长度为 arity
    double (*eval)(const double* args);
    CodegenSpec codegen;
    // 可选规范化改写：设置后 func() 构造直接返回 rewrite(args)，
    // 不再产生 Func 节点（如 sqrt(x) → x^(1/2)，保证同底幂可合并）。
    std::function<ExprPtr(const std::vector<ExprPtr>& args)> rewrite = nullptr;
};

class FuncRegistry {
public:
    // 首次调用时完成内置函数注册（sin cos tan asin acos atan
    // sinh cosh tanh exp log sqrt abs sign）
    static FuncRegistry& instance();

    FuncId add(FuncDef def);  // 重名抛 Error
    const FuncDef& get(FuncId id) const;
    std::optional<FuncId> find(std::string_view name) const;
    size_t size() const { return defs_.size(); }

private:
    FuncRegistry();
    std::vector<FuncDef> defs_;
    std::vector<std::string> names_;  // 与 defs_ 同步，用于查找
};

}  // namespace frontier
