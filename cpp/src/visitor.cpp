#include "frontier/visitor.hpp"

#include <unordered_set>

namespace frontier {

namespace {

void collect(const ExprPtr& e, std::unordered_set<const Expr*>& seen,
             std::vector<ExprPtr>& out) {
    if (!seen.insert(e.get()).second) return;
    if (e->kind() == ExprKind::Symbol) {
        out.push_back(e);
        return;
    }
    for_each_child(*e, [&](const ExprPtr& c) { collect(c, seen, out); });
}

}  // namespace

std::vector<ExprPtr> free_symbols(const ExprPtr& e) {
    std::unordered_set<const Expr*> seen;
    std::vector<ExprPtr> out;
    collect(e, seen, out);
    return out;
}

}  // namespace frontier
