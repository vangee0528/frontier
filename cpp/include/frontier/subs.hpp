#pragma once

#include <utility>
#include <vector>

#include "frontier/expr.hpp"

namespace frontier {

// 子表达式替换：把 e 中每个与 map 键结构相等的子表达式替换为对应值。
// 键可以是任意表达式（不限于 Symbol）——interning 使匹配为 O(1) 指针比较。
// 自底向上重建，结果经 builders 自动规范化（如 subs(x+y, {y:-x}) → 0）。
ExprPtr subs(const ExprPtr& e,
             const std::vector<std::pair<ExprPtr, ExprPtr>>& map);

}  // namespace frontier
