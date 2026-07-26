#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "frontier/number.hpp"

namespace frontier {

class Expr;
using ExprPtr = std::shared_ptr<const Expr>;

// 注册表中数学函数的稳定编号（见 func_registry.hpp）
using FuncId = uint16_t;

// 表达式节点种类。结构性且封闭：数学函数一律走 Func + FuncRegistry，
// 不为每个函数新增节点类。新增结构种类（如 Matrix）需同步扩展
// compare/hash/visitor/printer/diff/tape，见 docs/internals.md。
enum class ExprKind : uint8_t { Constant, Symbol, Add, Mul, Pow, Func };

// Add 的项：coeff * term
struct AddTerm {
    ExprPtr term;
    Number coeff;
};

// Mul 的因子：base ^ exp
struct MulFactor {
    ExprPtr base;
    ExprPtr exp;
};

// 不可变表达式节点。
//
// 规范形不变量（只能经 builders.hpp 构造，detail::make_* 仅做 intern）：
//  - Add：terms 非空且按 compare(term) 升序；term 的 kind ∉ {Constant, Add}，
//    若为 Mul 则其 coeff==1；每个 coeff 非零；
//    不存在「单项 + coeff==0 + 项系数==1」的退化形（应坍缩为该项本身）；
//    不存在「单项 + coeff==0」（应表示为 Mul）。
//  - Mul：factors 非空且按 compare(base) 升序；base 为 Mul 时 exp 必为
//    非整数（整数指数已在构造期分配展开；符号/分数指数保持嵌套，
//    如 (x·y)^z）；exp 不为 Constant(0)；coeff 非零；
//    不存在「coeff==1 + 单因子」的退化形（应坍缩为 Pow 或 base）。
//  - Pow：exp ∉ {Constant(0), Constant(1)}；可精确折叠的常量幂已折叠。
//
// 所有节点经全局 intern 表去重：指针相等 ⇔ 结构相等。
class Expr : public std::enable_shared_from_this<Expr> {
public:
    struct ConstantData { Number value; };
    struct SymbolData { std::string name; };
    struct AddData { Number coeff; std::vector<AddTerm> terms; };
    struct MulData { Number coeff; std::vector<MulFactor> factors; };
    struct PowData { ExprPtr base; ExprPtr exp; };
    struct FuncData { FuncId id; std::vector<ExprPtr> args; };

    using Payload = std::variant<ConstantData, SymbolData, AddData, MulData,
                                 PowData, FuncData>;

    ExprKind kind() const { return kind_; }
    size_t hash() const { return hash_; }

    // 载荷访问器（断言 kind 匹配）
    const Number& number() const { return std::get<ConstantData>(payload_).value; }
    const std::string& name() const { return std::get<SymbolData>(payload_).name; }
    const Number& add_coeff() const { return std::get<AddData>(payload_).coeff; }
    const std::vector<AddTerm>& terms() const { return std::get<AddData>(payload_).terms; }
    const Number& mul_coeff() const { return std::get<MulData>(payload_).coeff; }
    const std::vector<MulFactor>& factors() const { return std::get<MulData>(payload_).factors; }
    const ExprPtr& base() const { return std::get<PowData>(payload_).base; }
    const ExprPtr& exp() const { return std::get<PowData>(payload_).exp; }
    FuncId func_id() const { return std::get<FuncData>(payload_).id; }
    const std::vector<ExprPtr>& args() const { return std::get<FuncData>(payload_).args; }

    bool is_constant() const { return kind_ == ExprKind::Constant; }
    bool is_zero() const { return is_constant() && number().is_zero(); }
    bool is_one() const { return is_constant() && number().is_one(); }

    // 规范全序：先 kind 序，再载荷递归比较。返回 <0 / 0 / >0。
    // interning 保证同对象指针相等，比较从指针相等快速路径开始。
    static int compare(const ExprPtr& a, const ExprPtr& b);

    ~Expr() = default;
    Expr(const Expr&) = delete;
    Expr& operator=(const Expr&) = delete;

private:
    friend class Interner;
    Expr(ExprKind kind, Payload payload, size_t hash)
        : kind_(kind), payload_(std::move(payload)), hash_(hash) {}

    ExprKind kind_;
    Payload payload_;
    size_t hash_;
};

// 内部 interning 构造入口：假定载荷已满足规范形不变量（debug 下断言），
// 仅负责去重。上层代码一律使用 builders.hpp。
namespace detail {
ExprPtr make_constant(Number value);
ExprPtr make_symbol(std::string name);
ExprPtr make_add(Number coeff, std::vector<AddTerm> terms);
ExprPtr make_mul(Number coeff, std::vector<MulFactor> factors);
ExprPtr make_pow(ExprPtr base, ExprPtr exp);
ExprPtr make_func(FuncId id, std::vector<ExprPtr> args);
}  // namespace detail

}  // namespace frontier
