#pragma once

#include "frontier/expr.hpp"

namespace frontier {

// 对符号 var 求偏导。var 必须是 Symbol 节点，否则抛 Error。
// DAG 上 memo 化：共享子表达式只求导一次；结果经 builders 自动规范化。
ExprPtr diff(const ExprPtr& e, const ExprPtr& var);

}  // namespace frontier
