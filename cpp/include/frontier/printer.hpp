#pragma once

#include <string>

#include "frontier/expr.hpp"

namespace frontier {

// 中缀打印（确定性，最小括号），用于 repr 与测试快照。
std::string to_string(const ExprPtr& e);

}  // namespace frontier
