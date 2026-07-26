#pragma once

#include <stdexcept>

namespace frontier {

// 所有 Frontier 异常的基类；绑定层将其映射为 Python 的 FrontierError。
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// 数学域错误（除零、非法有理数等）。
struct DomainError : Error {
    using Error::Error;
};

// 降级 / 代码生成阶段错误。
struct CompileError : Error {
    using Error::Error;
};

}  // namespace frontier
