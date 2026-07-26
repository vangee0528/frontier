#pragma once

#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "frontier/expr.hpp"

namespace frontier {

// 表达式的唯一公共构造入口。全部规范化规则（扁平化、同类项/同底数合并、
// 常量折叠、单位元/零元消去）集中于此，保证任何可达 Expr 均为规范形。
//
// 精确性规则：只有当操作数含 Real 时才允许产生浮点折叠；
// 纯精确输入若无法精确折叠（如 2^(1/2)、sin(0)），保持符号形式。

ExprPtr symbol(std::string name);
ExprPtr constant(Number n);
ExprPtr integer(int64_t v);
ExprPtr rational(int64_t num, int64_t den);
ExprPtr real(double v);

const ExprPtr& zero();
const ExprPtr& one();
const ExprPtr& minus_one();

ExprPtr add(std::span<const ExprPtr> ops);
ExprPtr add(const ExprPtr& a, const ExprPtr& b);
inline ExprPtr add(std::initializer_list<ExprPtr> ops) {
    return add(std::span<const ExprPtr>(ops.begin(), ops.size()));
}
ExprPtr sub(const ExprPtr& a, const ExprPtr& b);
ExprPtr neg(const ExprPtr& a);

ExprPtr mul(std::span<const ExprPtr> ops);
ExprPtr mul(const ExprPtr& a, const ExprPtr& b);
inline ExprPtr mul(std::initializer_list<ExprPtr> ops) {
    return mul(std::span<const ExprPtr>(ops.begin(), ops.size()));
}
ExprPtr div(const ExprPtr& a, const ExprPtr& b);

ExprPtr pow(const ExprPtr& base, const ExprPtr& exp);

ExprPtr func(FuncId id, std::vector<ExprPtr> args);
ExprPtr func(std::string_view name, std::vector<ExprPtr> args);  // 未注册抛 Error

}  // namespace frontier
