#include "frontier/tape.hpp"

#include <bit>
#include <unordered_map>
#include <unordered_set>

#include "frontier/error.hpp"
#include "frontier/printer.hpp"
#include "frontier/visitor.hpp"

namespace frontier {

namespace {

class Lowerer {
public:
    explicit Lowerer(std::span<const ExprPtr> args) {
        tape_.num_inputs = args.size();
        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i]->kind() != ExprKind::Symbol)
                throw CompileError("lower: argument " + std::to_string(i) +
                                   " is not a Symbol");
            TapeOp op;
            op.kind = TapeOp::Kind::Input;
            op.input_index = static_cast<int32_t>(i);
            memo_[args[i].get()] = push(std::move(op));
        }
    }

    int32_t emit(const ExprPtr& e) {
        if (auto it = memo_.find(e.get()); it != memo_.end()) return it->second;
        const int32_t v = compute(e);
        memo_.emplace(e.get(), v);
        return v;
    }

    void add_output(int32_t v) { tape_.outputs.push_back(v); }
    Tape take() && { return std::move(tape_); }

private:
    int32_t push(TapeOp op) {
        tape_.ops.push_back(std::move(op));
        return static_cast<int32_t>(tape_.ops.size() - 1);
    }

    int32_t emit_const(double v) {
        const uint64_t bits = std::bit_cast<uint64_t>(v);
        if (auto it = const_memo_.find(bits); it != const_memo_.end()) return it->second;
        TapeOp op;
        op.kind = TapeOp::Kind::Const;
        op.cval = v;
        const int32_t id = push(std::move(op));
        const_memo_.emplace(bits, id);
        return id;
    }

    int32_t emit_bin(TapeOp::Kind k, int32_t a, int32_t b) {
        TapeOp op;
        op.kind = k;
        op.a = a;
        op.b = b;
        return push(std::move(op));
    }

    // bᵢ^eᵢ 因子 / Pow 节点共用
    int32_t emit_pow(const ExprPtr& base, const ExprPtr& exp) {
        const int32_t bv = emit(base);
        if (exp->is_one()) return bv;
        return emit_bin(TapeOp::Kind::Pow, bv, emit(exp));
    }

    int32_t compute(const ExprPtr& e) {
        switch (e->kind()) {
            case ExprKind::Constant:
                return emit_const(e->number().to_double());
            case ExprKind::Symbol:
                // args 中的符号已在构造时登记，走到这里即自由变量缺失
                throw CompileError("lower: free symbol '" + e->name() +
                                   "' is not in the argument list");
            case ExprKind::Add: {
                int32_t acc = -1;
                for (const auto& [t, c] : e->terms()) {
                    int32_t tv = emit(t);
                    if (!c.is_one())
                        tv = emit_bin(TapeOp::Kind::Mul, emit_const(c.to_double()), tv);
                    acc = (acc < 0) ? tv : emit_bin(TapeOp::Kind::Add, acc, tv);
                }
                if (!e->add_coeff().is_zero() || acc < 0) {
                    const int32_t cv = emit_const(e->add_coeff().to_double());
                    acc = (acc < 0) ? cv : emit_bin(TapeOp::Kind::Add, acc, cv);
                }
                return acc;
            }
            case ExprKind::Mul: {
                int32_t acc = -1;
                for (const auto& [b, ex] : e->factors()) {
                    const int32_t fv = emit_pow(b, ex);
                    acc = (acc < 0) ? fv : emit_bin(TapeOp::Kind::Mul, acc, fv);
                }
                if (!e->mul_coeff().is_one() || acc < 0) {
                    const int32_t cv = emit_const(e->mul_coeff().to_double());
                    acc = (acc < 0) ? cv : emit_bin(TapeOp::Kind::Mul, cv, acc);
                }
                return acc;
            }
            case ExprKind::Pow:
                return emit_pow(e->base(), e->exp());
            case ExprKind::Func: {
                TapeOp op;
                op.kind = TapeOp::Kind::Call;
                op.func = e->func_id();
                op.args.reserve(e->args().size());
                for (const auto& a : e->args()) op.args.push_back(emit(a));
                return push(std::move(op));
            }
        }
        throw CompileError("lower: unsupported expression: " + to_string(e));
    }

    Tape tape_;
    std::unordered_map<const Expr*, int32_t> memo_;
    std::unordered_map<uint64_t, int32_t> const_memo_;
};

}  // namespace

Tape lower(std::span<const ExprPtr> exprs, std::span<const ExprPtr> args) {
    // 预检：先验证实参均为 Symbol，再一次性列出全部缺失的自由符号
    {
        for (size_t i = 0; i < args.size(); ++i)
            if (args[i]->kind() != ExprKind::Symbol)
                throw CompileError("lower: argument " + std::to_string(i) +
                                   " is not a Symbol");
        std::unordered_set<const Expr*> in_args;
        for (const auto& a : args) in_args.insert(a.get());
        std::vector<std::string> missing;
        std::unordered_set<const Expr*> reported;
        for (const auto& e : exprs)
            for (const auto& s : free_symbols(e))
                if (!in_args.count(s.get()) && reported.insert(s.get()).second)
                    missing.push_back(s->name());
        if (!missing.empty()) {
            std::string msg = "lower: free symbol(s) not in the argument list: ";
            for (size_t i = 0; i < missing.size(); ++i)
                msg += (i ? ", '" : "'") + missing[i] + "'";
            msg += ". Arguments given: [";
            for (size_t i = 0; i < args.size(); ++i)
                msg += (i ? ", " : "") + args[i]->name();
            msg += "]";
            throw CompileError(msg);
        }
    }
    Lowerer lw(args);
    for (const auto& e : exprs) lw.add_output(lw.emit(e));
    return std::move(lw).take();
}

}  // namespace frontier
