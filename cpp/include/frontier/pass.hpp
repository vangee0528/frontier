#pragma once

#include <memory>
#include <string>
#include <vector>

#include "frontier/expr.hpp"

namespace frontier {

// 显式化简 Pass 接口。构造期规范化（builders）承担 v1 的全部化简；
// 本管线是未来重量级变换（三角化简、多项式展开/合并、基于假设的
// 重写等）的插槽，见 docs/internals.md。
class ExprPass {
public:
    virtual ~ExprPass() = default;
    virtual std::string name() const = 0;
    virtual ExprPtr run(const ExprPtr& e) const = 0;
};

class PassPipeline {
public:
    void append(std::unique_ptr<ExprPass> pass) { passes_.push_back(std::move(pass)); }

    ExprPtr run(ExprPtr e) const {
        for (const auto& p : passes_) e = p->run(e);
        return e;
    }

    size_t size() const { return passes_.size(); }

    // v1 默认管线为空：规范化已在构造期完成
    static PassPipeline default_pipeline() { return {}; }

private:
    std::vector<std::unique_ptr<ExprPass>> passes_;
};

}  // namespace frontier
