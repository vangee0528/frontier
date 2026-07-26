#include "frontier/subs.hpp"

#include <unordered_map>

#include "frontier/builders.hpp"

namespace frontier {

namespace {

class Substitutor {
public:
    explicit Substitutor(const std::vector<std::pair<ExprPtr, ExprPtr>>& map) {
        for (const auto& [k, v] : map) map_.emplace(k.get(), v);
    }

    ExprPtr run(const ExprPtr& e) {
        if (auto it = map_.find(e.get()); it != map_.end()) return it->second;
        if (auto it = memo_.find(e.get()); it != memo_.end()) return it->second;
        ExprPtr r = rebuild(e);
        memo_.emplace(e.get(), r);
        return r;
    }

private:
    ExprPtr rebuild(const ExprPtr& e) {
        switch (e->kind()) {
            case ExprKind::Constant:
            case ExprKind::Symbol:
                return e;
            case ExprKind::Add: {
                std::vector<ExprPtr> parts;
                parts.reserve(e->terms().size() + 1);
                parts.push_back(constant(e->add_coeff()));
                for (const auto& [t, c] : e->terms())
                    parts.push_back(mul(constant(c), run(t)));
                return add(std::span<const ExprPtr>(parts));
            }
            case ExprKind::Mul: {
                std::vector<ExprPtr> parts;
                parts.reserve(e->factors().size() + 1);
                parts.push_back(constant(e->mul_coeff()));
                for (const auto& [b, ex] : e->factors())
                    parts.push_back(pow(run(b), run(ex)));
                return mul(std::span<const ExprPtr>(parts));
            }
            case ExprKind::Pow:
                return pow(run(e->base()), run(e->exp()));
            case ExprKind::Func: {
                std::vector<ExprPtr> args;
                args.reserve(e->args().size());
                for (const auto& a : e->args()) args.push_back(run(a));
                return func(e->func_id(), std::move(args));
            }
        }
        return e;
    }

    std::unordered_map<const Expr*, ExprPtr> map_;
    std::unordered_map<const Expr*, ExprPtr> memo_;
};

}  // namespace

ExprPtr subs(const ExprPtr& e,
             const std::vector<std::pair<ExprPtr, ExprPtr>>& map) {
    return Substitutor(map).run(e);
}

}  // namespace frontier
