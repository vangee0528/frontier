#include "frontier/builders.hpp"

#include <algorithm>
#include <unordered_map>

#include "frontier/error.hpp"
#include "frontier/func_registry.hpp"

namespace frontier {

namespace {

// 把 c*t（t 非 Constant / 非 Add）物化为规范节点
ExprPtr scale_term(const ExprPtr& t, const Number& c);

// Mul 节点去掉数值系数后重建
ExprPtr mul_without_coeff(const Expr& m) {
    const auto& fs = m.factors();
    if (fs.size() == 1) return pow(fs[0].base, fs[0].exp);
    return detail::make_mul(Number::integer(1), fs);
}

// e 拆为 (数值系数, 剩余部分)；纯常量时剩余部分为 nullptr
std::pair<Number, ExprPtr> as_coeff_rest(const ExprPtr& e) {
    switch (e->kind()) {
        case ExprKind::Constant: return {e->number(), nullptr};
        case ExprKind::Mul: return {e->mul_coeff(), mul_without_coeff(*e)};
        default: return {Number::integer(1), e};
    }
}

ExprPtr scale_term(const ExprPtr& t, const Number& c) {
    if (c.is_one()) return t;
    if (t->kind() == ExprKind::Mul)  // 不变量：Add 项中的 Mul 系数为 1
        return detail::make_mul(c, t->factors());
    std::vector<MulFactor> fs;
    if (t->kind() == ExprKind::Pow)
        fs.push_back({t->base(), t->exp()});
    else
        fs.push_back({t, one()});
    return detail::make_mul(c, std::move(fs));
}

}  // namespace

// ---------------------------------------------------------------------------
// 叶子
// ---------------------------------------------------------------------------

ExprPtr symbol(std::string name) {
    if (name.empty()) throw Error("symbol name must be non-empty");
    return detail::make_symbol(std::move(name));
}

ExprPtr constant(Number n) { return detail::make_constant(std::move(n)); }
ExprPtr integer(int64_t v) { return constant(Number::integer(v)); }
ExprPtr rational(int64_t num, int64_t den) { return constant(Number::rational(num, den)); }
ExprPtr real(double v) { return constant(Number::real(v)); }

// 常量单例刻意泄漏（同 intern 表策略）：若持有真正的静态 ExprPtr，
// CRT 静态析构阶段其 Deleter 会在 thread_local 队列已析构后运行（UB，
// 曾导致进程退出段错误）。泄漏三个节点无任何代价。
const ExprPtr& zero() {
    static const ExprPtr* e = new ExprPtr(integer(0));
    return *e;
}
const ExprPtr& one() {
    static const ExprPtr* e = new ExprPtr(integer(1));
    return *e;
}
const ExprPtr& minus_one() {
    static const ExprPtr* e = new ExprPtr(integer(-1));
    return *e;
}

// ---------------------------------------------------------------------------
// Add：coeff + Σ cᵢ·tᵢ，构造期完成同类项合并与常量折叠
// ---------------------------------------------------------------------------

ExprPtr add(std::span<const ExprPtr> ops) {
    Number coeff = Number::integer(0);
    std::vector<AddTerm> acc;
    std::unordered_map<const Expr*, size_t> index;

    auto accumulate = [&](const ExprPtr& t, const Number& c) {
        auto [it, inserted] = index.try_emplace(t.get(), acc.size());
        if (inserted)
            acc.push_back({t, c});
        else
            acc[it->second].coeff = acc[it->second].coeff + c;
    };

    for (const ExprPtr& op : ops) {
        switch (op->kind()) {
            case ExprKind::Constant:
                coeff = coeff + op->number();
                break;
            case ExprKind::Add:
                coeff = coeff + op->add_coeff();
                for (const auto& [t, c] : op->terms()) accumulate(t, c);
                break;
            default: {
                auto [c, rest] = as_coeff_rest(op);
                if (rest == nullptr)
                    coeff = coeff + c;
                else
                    accumulate(rest, c);
            }
        }
    }

    std::erase_if(acc, [](const AddTerm& t) { return t.coeff.is_zero(); });

    if (acc.empty()) return constant(coeff);
    if (acc.size() == 1 && coeff.is_zero())
        return scale_term(acc[0].term, acc[0].coeff);

    std::sort(acc.begin(), acc.end(), [](const AddTerm& a, const AddTerm& b) {
        return Expr::compare(a.term, b.term) < 0;
    });
    return detail::make_add(std::move(coeff), std::move(acc));
}

ExprPtr add(const ExprPtr& a, const ExprPtr& b) {
    const ExprPtr ops[] = {a, b};
    return add(std::span<const ExprPtr>(ops));
}

ExprPtr neg(const ExprPtr& a) { return mul(minus_one(), a); }
ExprPtr sub(const ExprPtr& a, const ExprPtr& b) { return add(a, neg(b)); }

// ---------------------------------------------------------------------------
// Mul：coeff · Π bᵢ^eᵢ，构造期完成同底数合并与常量折叠
// ---------------------------------------------------------------------------

ExprPtr mul(std::span<const ExprPtr> ops) {
    Number coeff = Number::integer(1);
    std::vector<MulFactor> acc;
    std::unordered_map<const Expr*, size_t> index;

    auto accumulate = [&](const ExprPtr& b, const ExprPtr& e) {
        auto [it, inserted] = index.try_emplace(b.get(), acc.size());
        if (inserted)
            acc.push_back({b, e});
        else
            acc[it->second].exp = add(acc[it->second].exp, e);  // 同底数：指数相加（符号加法）
    };

    for (const ExprPtr& op : ops) {
        switch (op->kind()) {
            case ExprKind::Constant:
                coeff = coeff * op->number();
                break;
            case ExprKind::Mul:
                coeff = coeff * op->mul_coeff();
                for (const auto& [b, e] : op->factors()) accumulate(b, e);
                break;
            case ExprKind::Pow:
                accumulate(op->base(), op->exp());
                break;
            default:
                accumulate(op, one());
        }
    }

    if (coeff.is_zero()) return zero();

    // 逐因子后处理：b^0 消去、常量幂折叠（遵守精确性规则）、1^x 消去
    std::vector<MulFactor> kept;
    kept.reserve(acc.size());
    for (auto& f : acc) {
        if (f.exp->is_zero()) continue;
        if (f.base->kind() == ExprKind::Constant) {
            const Number& bn = f.base->number();
            if (bn.is_one()) continue;
            if (f.exp->kind() == ExprKind::Constant) {
                const Number& en = f.exp->number();
                auto r = Number::pow(bn, en);
                if (r && (r->is_exact() || !bn.is_exact() || !en.is_exact())) {
                    coeff = coeff * *r;
                    continue;
                }
            }
        }
        kept.push_back(std::move(f));
    }

    if (coeff.is_zero()) return zero();
    if (kept.empty()) return constant(coeff);

    // 二次归一：同底指数相加可能把「Mul/Pow 底 + 符号指数」的合法嵌套
    // 变成「Mul/Pow 底 + 整数指数」——该形态必须经 pow() 分配展开
    // （如 (-x)^(1/2)·(-x)^(-3/2) → (-x)^(-1) → -x^(-1)，否则丢符号）。
    // pow() 对整数指数的 Mul/Pow 底恒展开，递归一层即收敛。
    for (const auto& f : kept) {
        const bool int_exp = f.exp->kind() == ExprKind::Constant &&
                             f.exp->number().is_int();
        if (int_exp && (f.base->kind() == ExprKind::Mul ||
                        f.base->kind() == ExprKind::Pow)) {
            std::vector<ExprPtr> parts;
            parts.reserve(kept.size() + 1);
            parts.push_back(constant(coeff));
            for (const auto& g : kept) parts.push_back(pow(g.base, g.exp));
            return mul(std::span<const ExprPtr>(parts));
        }
    }

    if (kept.size() == 1 && coeff.is_one()) return pow(kept[0].base, kept[0].exp);

    // 数值系数对单个 Add 因子分配律展开：2*(x+y) → 2x+2y，利于同类项合并
    if (kept.size() == 1 && kept[0].exp->is_one() &&
        kept[0].base->kind() == ExprKind::Add) {
        const Expr& a = *kept[0].base;
        std::vector<AddTerm> terms = a.terms();
        for (auto& t : terms) t.coeff = t.coeff * coeff;
        return detail::make_add(a.add_coeff() * coeff, std::move(terms));
    }

    std::sort(kept.begin(), kept.end(), [](const MulFactor& a, const MulFactor& b) {
        return Expr::compare(a.base, b.base) < 0;
    });
    return detail::make_mul(std::move(coeff), std::move(kept));
}

ExprPtr mul(const ExprPtr& a, const ExprPtr& b) {
    const ExprPtr ops[] = {a, b};
    return mul(std::span<const ExprPtr>(ops));
}

ExprPtr div(const ExprPtr& a, const ExprPtr& b) { return mul(a, pow(b, minus_one())); }

// ---------------------------------------------------------------------------
// Pow
// ---------------------------------------------------------------------------

ExprPtr pow(const ExprPtr& base, const ExprPtr& exp) {
    if (exp->kind() == ExprKind::Constant) {
        const Number& n = exp->number();
        if (n.is_zero()) return one();   // 注：0^0 按约定折叠为 1
        if (n.is_one()) return base;

        if (base->kind() == ExprKind::Constant) {
            const Number& bn = base->number();
            auto r = Number::pow(bn, n);
            // 精确性规则：纯精确输入只接受精确折叠（sqrt(2) 保持符号）；
            // 含 Real 输入总是折叠；负底数非整指数（nullopt）保持符号。
            if (r && (r->is_exact() || !bn.is_exact() || !n.is_exact()))
                return constant(*r);
            return detail::make_pow(base, exp);
        }

        if (n.is_int()) {
            if (base->kind() == ExprKind::Mul) {
                // (c·Πbᵢ^eᵢ)^n → c^n · Π bᵢ^(eᵢ·n)，仅整数指数下安全
                std::vector<ExprPtr> parts;
                parts.reserve(base->factors().size() + 1);
                parts.push_back(pow(constant(base->mul_coeff()), exp));
                for (const auto& [b, e] : base->factors())
                    parts.push_back(pow(b, mul(e, exp)));
                return mul(std::span<const ExprPtr>(parts));
            }
            if (base->kind() == ExprKind::Pow)
                return pow(base->base(), mul(base->exp(), exp));  // (b^e)^n → b^(e·n)
        }
        return detail::make_pow(base, exp);
    }

    // 符号指数
    if (base->is_one()) return one();
    return detail::make_pow(base, exp);
}

// ---------------------------------------------------------------------------
// Func
// ---------------------------------------------------------------------------

ExprPtr func(FuncId id, std::vector<ExprPtr> args) {
    const FuncDef& def = FuncRegistry::instance().get(id);
    if (static_cast<int>(args.size()) != def.arity)
        throw Error("function '" + def.name + "' expects " +
                    std::to_string(def.arity) + " argument(s), got " +
                    std::to_string(args.size()));

    if (def.rewrite) return def.rewrite(args);

    // 常量折叠：仅当所有实参均为 Real 常量（保持精确性规则：sin(0) 不折叠）
    bool all_real = true;
    for (const auto& a : args) {
        if (!(a->kind() == ExprKind::Constant && a->number().kind() == Number::Kind::Real)) {
            all_real = false;
            break;
        }
    }
    if (all_real && def.eval != nullptr) {
        std::vector<double> vals;
        vals.reserve(args.size());
        for (const auto& a : args) vals.push_back(a->number().to_double());
        return real(def.eval(vals.data()));
    }
    return detail::make_func(id, std::move(args));
}

ExprPtr func(std::string_view name, std::vector<ExprPtr> args) {
    auto id = FuncRegistry::instance().find(name);
    if (!id) throw Error("unknown function: " + std::string(name));
    return func(*id, std::move(args));
}

}  // namespace frontier
