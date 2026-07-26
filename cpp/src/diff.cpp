#include "frontier/diff.hpp"

#include <unordered_map>

#include "frontier/builders.hpp"
#include "frontier/error.hpp"
#include "frontier/func_registry.hpp"

namespace frontier {

namespace {

class Differentiator {
public:
    explicit Differentiator(const Expr* var) : var_(var) {}

    ExprPtr run(const ExprPtr& e) {
        if (auto it = memo_.find(e.get()); it != memo_.end()) return it->second;
        ExprPtr d = compute(e);
        memo_.emplace(e.get(), d);
        return d;
    }

private:
    // d(b^e)：常量指数走幂法则，一般情形 b^e·(e'·ln b + e·b'/b)
    ExprPtr d_pow(const ExprPtr& b, const ExprPtr& e) {
        const ExprPtr db = run(b);
        if (e->kind() == ExprKind::Constant) {
            if (db->is_zero()) return zero();
            // n·b^(n-1)·b'
            return mul({e, pow(b, constant(e->number() - Number::integer(1))), db});
        }
        const ExprPtr de = run(e);
        std::vector<ExprPtr> terms;
        if (!de->is_zero()) terms.push_back(mul(de, func("log", {b})));
        if (!db->is_zero()) terms.push_back(mul({e, db, pow(b, minus_one())}));
        if (terms.empty()) return zero();
        const ExprPtr inner = terms.size() == 1 ? terms[0] : add(terms[0], terms[1]);
        return mul(pow(b, e), inner);
    }

    ExprPtr compute(const ExprPtr& e) {
        switch (e->kind()) {
            case ExprKind::Constant:
                return zero();
            case ExprKind::Symbol:
                return e.get() == var_ ? one() : zero();
            case ExprKind::Add: {
                std::vector<ExprPtr> parts;
                parts.reserve(e->terms().size());
                for (const auto& [t, c] : e->terms())
                    parts.push_back(mul(constant(c), run(t)));
                return add(std::span<const ExprPtr>(parts));
            }
            case ExprKind::Mul: {
                // 乘积法则：Σᵢ coeff·(Π_{j≠i} bⱼ^eⱼ)·d(bᵢ^eᵢ)
                const auto& fs = e->factors();
                std::vector<ExprPtr> sum_parts;
                for (size_t i = 0; i < fs.size(); ++i) {
                    const ExprPtr dpi = d_pow(fs[i].base, fs[i].exp);
                    if (dpi->is_zero()) continue;
                    std::vector<ExprPtr> prod;
                    prod.reserve(fs.size() + 1);
                    prod.push_back(constant(e->mul_coeff()));
                    for (size_t j = 0; j < fs.size(); ++j)
                        if (j != i) prod.push_back(pow(fs[j].base, fs[j].exp));
                    prod.push_back(dpi);
                    sum_parts.push_back(mul(std::span<const ExprPtr>(prod)));
                }
                return add(std::span<const ExprPtr>(sum_parts));
            }
            case ExprKind::Pow:
                return d_pow(e->base(), e->exp());
            case ExprKind::Func: {
                // 链式法则：Σᵢ ∂f/∂argᵢ · d(argᵢ)
                const FuncDef& def = FuncRegistry::instance().get(e->func_id());
                std::vector<ExprPtr> parts;
                for (size_t i = 0; i < e->args().size(); ++i) {
                    const ExprPtr da = run(e->args()[i]);
                    if (da->is_zero()) continue;
                    parts.push_back(mul(def.deriv(e->args(), static_cast<int>(i)), da));
                }
                return add(std::span<const ExprPtr>(parts));
            }
        }
        return zero();
    }

    const Expr* var_;
    std::unordered_map<const Expr*, ExprPtr> memo_;
};

}  // namespace

ExprPtr diff(const ExprPtr& e, const ExprPtr& var) {
    if (var->kind() != ExprKind::Symbol)
        throw Error("diff: differentiation variable must be a Symbol");
    return Differentiator(var.get()).run(e);
}

}  // namespace frontier
