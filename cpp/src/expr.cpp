#include "frontier/expr.hpp"

#include <cassert>
#include <mutex>
#include <unordered_map>

namespace frontier {

namespace {

size_t hash_combine(size_t seed, size_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

int cmp_int(int64_t a, int64_t b) { return a < b ? -1 : (a > b ? 1 : 0); }

}  // namespace

int Expr::compare(const ExprPtr& a, const ExprPtr& b) {
    if (a.get() == b.get()) return 0;  // interning：指针相等 ⇔ 结构相等
    if (a->kind_ != b->kind_)
        return static_cast<int>(a->kind_) < static_cast<int>(b->kind_) ? -1 : 1;
    switch (a->kind_) {
        case ExprKind::Constant:
            return Number::compare(a->number(), b->number());
        case ExprKind::Symbol: {
            const int c = a->name().compare(b->name());
            return c < 0 ? -1 : (c > 0 ? 1 : 0);
        }
        case ExprKind::Add: {
            const auto& ta = a->terms();
            const auto& tb = b->terms();
            const size_t n = std::min(ta.size(), tb.size());
            for (size_t i = 0; i < n; ++i) {
                if (int c = compare(ta[i].term, tb[i].term)) return c;
                if (int c = Number::compare(ta[i].coeff, tb[i].coeff)) return c;
            }
            if (int c = cmp_int((int64_t)ta.size(), (int64_t)tb.size())) return c;
            return Number::compare(a->add_coeff(), b->add_coeff());
        }
        case ExprKind::Mul: {
            const auto& fa = a->factors();
            const auto& fb = b->factors();
            const size_t n = std::min(fa.size(), fb.size());
            for (size_t i = 0; i < n; ++i) {
                if (int c = compare(fa[i].base, fb[i].base)) return c;
                if (int c = compare(fa[i].exp, fb[i].exp)) return c;
            }
            if (int c = cmp_int((int64_t)fa.size(), (int64_t)fb.size())) return c;
            return Number::compare(a->mul_coeff(), b->mul_coeff());
        }
        case ExprKind::Pow: {
            if (int c = compare(a->base(), b->base())) return c;
            return compare(a->exp(), b->exp());
        }
        case ExprKind::Func: {
            if (int c = cmp_int(a->func_id(), b->func_id())) return c;
            const auto& xa = a->args();
            const auto& xb = b->args();
            const size_t n = std::min(xa.size(), xb.size());
            for (size_t i = 0; i < n; ++i)
                if (int c = compare(xa[i], xb[i])) return c;
            return cmp_int((int64_t)xa.size(), (int64_t)xb.size());
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Interner：全局 hash-consing 表。
// 弱引用桶 + 节点析构时自清理；互斥锁为粗粒度（若成瓶颈可换分片锁，接口不变）。
// ---------------------------------------------------------------------------
class Interner {
public:
    static ExprPtr intern(ExprKind kind, Expr::Payload payload) {
        const size_t h = compute_hash(kind, payload);
        // 声明先于 lock_guard：确保临时强引用在锁释放之后才析构，
        // 避免「恰好持有最后一个引用 → Deleter 重入取锁」的死锁
        std::vector<ExprPtr> keep_alive;
        std::lock_guard lk(table().mu);
        auto& bucket = table().buckets[h];
        for (auto it = bucket.begin(); it != bucket.end();) {
            if (auto sp = it->lock()) {
                if (shallow_equal(*sp, kind, payload)) return sp;
                keep_alive.push_back(std::move(sp));
                ++it;
            } else {
                it = bucket.erase(it);  // 死条目顺手清理
            }
        }
        ExprPtr sp(new Expr(kind, std::move(payload), h), Deleter{h});
        bucket.emplace_back(sp);
        return sp;
        // 注：重复命中时被丢弃的 payload（含子节点引用）在锁释放后才析构，
        // 子节点的 Deleter 不会在持锁状态下重入。
    }

private:
    struct Table {
        std::mutex mu;
        std::unordered_map<size_t, std::vector<std::weak_ptr<const Expr>>> buckets;
    };

    // 刻意泄漏：规避静态析构顺序问题（宿主进程退出时表无需清理）
    static Table& table() {
        static Table* t = new Table();
        return *t;
    }

    struct Deleter {
        size_t hash;
        void operator()(const Expr* p) const {
            {
                std::lock_guard lk(table().mu);
                auto it = table().buckets.find(hash);
                if (it != table().buckets.end()) {
                    auto& vec = it->second;
                    std::erase_if(vec, [](const std::weak_ptr<const Expr>& w) {
                        return w.expired();
                    });
                    if (vec.empty()) table().buckets.erase(it);
                }
            }
            // 迭代式析构：深表达式链（如 1e4 层嵌套）若走 shared_ptr
            // 递归释放会栈溢出。把载荷（持有子引用）移入线程局部队列，
            // 由最外层调用统一排空；重入的 Deleter 只入队不递归。
            // 队列对象按线程刻意泄漏：进程退出阶段（TLS 析构之后）仍可能
            // 有 Deleter 运行（如宿主语言运行时的迟释放），访问已析构的
            // thread_local 是 UB。每线程泄漏一个小对象，drain 后收缩容量。
            struct TeardownQueue {
                std::vector<Expr::Payload> q;
                bool draining = false;
            };
            thread_local TeardownQueue* tq = new TeardownQueue();
            tq->q.push_back(std::move(const_cast<Expr*>(p)->payload_));
            delete p;  // 载荷已移出，删除不再触达子节点
            if (tq->draining) return;
            tq->draining = true;
            while (!tq->q.empty()) {
                Expr::Payload payload = std::move(tq->q.back());
                tq->q.pop_back();
                // payload 在此析构，释放子引用；触发的 Deleter 仅入队不递归
            }
            tq->draining = false;
            tq->q.shrink_to_fit();
        }
    };

    static size_t compute_hash(ExprKind kind, const Expr::Payload& p) {
        size_t seed = static_cast<size_t>(kind) * 0x9e3779b97f4a7c15ULL;
        switch (kind) {
            case ExprKind::Constant:
                return hash_combine(seed, std::get<Expr::ConstantData>(p).value.hash());
            case ExprKind::Symbol:
                return hash_combine(seed,
                                    std::hash<std::string>{}(std::get<Expr::SymbolData>(p).name));
            case ExprKind::Add: {
                const auto& d = std::get<Expr::AddData>(p);
                seed = hash_combine(seed, d.coeff.hash());
                for (const auto& [t, c] : d.terms) {
                    seed = hash_combine(seed, t->hash());
                    seed = hash_combine(seed, c.hash());
                }
                return seed;
            }
            case ExprKind::Mul: {
                const auto& d = std::get<Expr::MulData>(p);
                seed = hash_combine(seed, d.coeff.hash());
                for (const auto& [b, e] : d.factors) {
                    seed = hash_combine(seed, b->hash());
                    seed = hash_combine(seed, e->hash());
                }
                return seed;
            }
            case ExprKind::Pow: {
                const auto& d = std::get<Expr::PowData>(p);
                seed = hash_combine(seed, d.base->hash());
                return hash_combine(seed, d.exp->hash());
            }
            case ExprKind::Func: {
                const auto& d = std::get<Expr::FuncData>(p);
                seed = hash_combine(seed, d.id);
                for (const auto& a : d.args) seed = hash_combine(seed, a->hash());
                return seed;
            }
        }
        return seed;
    }

    // 子节点已 intern ⇒ 浅比较（子节点用指针相等）即结构相等
    static bool shallow_equal(const Expr& e, ExprKind kind, const Expr::Payload& p) {
        if (e.kind() != kind) return false;
        switch (kind) {
            case ExprKind::Constant:
                return e.number() == std::get<Expr::ConstantData>(p).value;
            case ExprKind::Symbol:
                return e.name() == std::get<Expr::SymbolData>(p).name;
            case ExprKind::Add: {
                const auto& d = std::get<Expr::AddData>(p);
                if (!(e.add_coeff() == d.coeff)) return false;
                const auto& ts = e.terms();
                if (ts.size() != d.terms.size()) return false;
                for (size_t i = 0; i < ts.size(); ++i)
                    if (ts[i].term.get() != d.terms[i].term.get() ||
                        !(ts[i].coeff == d.terms[i].coeff))
                        return false;
                return true;
            }
            case ExprKind::Mul: {
                const auto& d = std::get<Expr::MulData>(p);
                if (!(e.mul_coeff() == d.coeff)) return false;
                const auto& fs = e.factors();
                if (fs.size() != d.factors.size()) return false;
                for (size_t i = 0; i < fs.size(); ++i)
                    if (fs[i].base.get() != d.factors[i].base.get() ||
                        fs[i].exp.get() != d.factors[i].exp.get())
                        return false;
                return true;
            }
            case ExprKind::Pow: {
                const auto& d = std::get<Expr::PowData>(p);
                return e.base().get() == d.base.get() && e.exp().get() == d.exp.get();
            }
            case ExprKind::Func: {
                const auto& d = std::get<Expr::FuncData>(p);
                if (e.func_id() != d.id) return false;
                const auto& as = e.args();
                if (as.size() != d.args.size()) return false;
                for (size_t i = 0; i < as.size(); ++i)
                    if (as[i].get() != d.args[i].get()) return false;
                return true;
            }
        }
        return false;
    }
};

namespace detail {

ExprPtr make_constant(Number value) {
    return Interner::intern(ExprKind::Constant, Expr::ConstantData{std::move(value)});
}

ExprPtr make_symbol(std::string name) {
    assert(!name.empty());
    return Interner::intern(ExprKind::Symbol, Expr::SymbolData{std::move(name)});
}

ExprPtr make_add(Number coeff, std::vector<AddTerm> terms) {
    assert(!terms.empty());
#ifndef NDEBUG
    for (size_t i = 0; i + 1 < terms.size(); ++i)
        assert(Expr::compare(terms[i].term, terms[i + 1].term) < 0);
    for (const auto& t : terms) {
        assert(!t.coeff.is_zero());
        assert(t.term->kind() != ExprKind::Constant && t.term->kind() != ExprKind::Add);
    }
    assert(!(terms.size() == 1 && coeff.is_zero()));
#endif
    return Interner::intern(ExprKind::Add, Expr::AddData{std::move(coeff), std::move(terms)});
}

ExprPtr make_mul(Number coeff, std::vector<MulFactor> factors) {
    assert(!factors.empty());
    assert(!coeff.is_zero());
#ifndef NDEBUG
    for (size_t i = 0; i + 1 < factors.size(); ++i)
        assert(Expr::compare(factors[i].base, factors[i + 1].base) < 0);
    for (const auto& f : factors) {
        // base 为 Mul/Pow 仅在「符号指数、不可安全分配」时合法：
        // (x·y)^z 或 (x^y)^z 作为因子保持嵌套（整数指数已在 pow() 分配）。
        // 同底合并经指针相等仍然有效，规范序不受影响。
        assert(!(f.base->kind() == ExprKind::Mul &&
                 f.exp->kind() == ExprKind::Constant &&
                 f.exp->number().is_int()));
        assert(!f.exp->is_zero());
    }
    assert(!(factors.size() == 1 && coeff.is_one()));
#endif
    return Interner::intern(ExprKind::Mul, Expr::MulData{std::move(coeff), std::move(factors)});
}

ExprPtr make_pow(ExprPtr base, ExprPtr exp) {
    assert(!exp->is_zero() && !exp->is_one());
    return Interner::intern(ExprKind::Pow, Expr::PowData{std::move(base), std::move(exp)});
}

ExprPtr make_func(FuncId id, std::vector<ExprPtr> args) {
    return Interner::intern(ExprKind::Func, Expr::FuncData{id, std::move(args)});
}

}  // namespace detail

}  // namespace frontier
