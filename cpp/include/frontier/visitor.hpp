#pragma once

#include <vector>

#include "frontier/expr.hpp"

namespace frontier {

// 按 kind 枚举直接子节点（不含 Number 载荷）。
// 所有遍历类变换（diff、tape 降级等）共用此原语。
template <class Fn>
void for_each_child(const Expr& e, Fn&& fn) {
    switch (e.kind()) {
        case ExprKind::Constant:
        case ExprKind::Symbol:
            break;
        case ExprKind::Add:
            for (const auto& t : e.terms()) fn(t.term);
            break;
        case ExprKind::Mul:
            for (const auto& f : e.factors()) {
                fn(f.base);
                fn(f.exp);
            }
            break;
        case ExprKind::Pow:
            fn(e.base());
            fn(e.exp());
            break;
        case ExprKind::Func:
            for (const auto& a : e.args()) fn(a);
            break;
    }
}

// 收集表达式中的自由符号（DAG 去重，按首次出现顺序）。
std::vector<ExprPtr> free_symbols(const ExprPtr& e);

}  // namespace frontier
