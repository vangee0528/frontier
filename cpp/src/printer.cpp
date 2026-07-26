#include "frontier/printer.hpp"

#include "frontier/func_registry.hpp"

namespace frontier {

namespace {

// 运算符优先级（越大结合越紧）
constexpr int kAdd = 10;
constexpr int kMul = 20;
constexpr int kPow = 30;
constexpr int kAtom = 40;

std::string print(const ExprPtr& e, int parent_prec);

std::string wrap(std::string s, bool need) {
    return need ? "(" + std::move(s) + ")" : std::move(s);
}

// 数值系数在乘法语境下的打印（有理数/负数在幂语境需要括号）
std::string print_number(const Number& n, int parent_prec) {
    const bool composite = n.kind() == Number::Kind::Rational || n.is_negative();
    return wrap(n.to_string(), composite && parent_prec > kMul);
}

// 单个 c·t 项
std::string print_term(const Number& coeff, const ExprPtr& term) {
    if (coeff.is_one()) return print(term, kMul);
    if (coeff.is_minus_one()) return "-" + print(term, kMul + 1);
    return print_number(coeff, kMul + 1) + "*" + print(term, kMul + 1);
}

// 以 " + "/" - " 拼接（负号吸收进连接符）
void join_into(std::string& out, const std::string& piece) {
    if (out.empty()) {
        out = piece;
        return;
    }
    if (!piece.empty() && piece[0] == '-')
        out += " - " + piece.substr(1);
    else
        out += " + " + piece;
}

std::string print(const ExprPtr& e, int parent_prec) {
    switch (e->kind()) {
        case ExprKind::Constant:
            return print_number(e->number(), parent_prec);
        case ExprKind::Symbol:
            return e->name();
        case ExprKind::Add: {
            std::string out;
            for (const auto& [t, c] : e->terms()) join_into(out, print_term(c, t));
            if (!e->add_coeff().is_zero()) join_into(out, e->add_coeff().to_string());
            return wrap(std::move(out), parent_prec > kAdd);
        }
        case ExprKind::Mul: {
            std::string out;
            const Number& c = e->mul_coeff();
            bool negate = false;
            if (c.is_minus_one()) {
                negate = true;
            } else if (!c.is_one()) {
                out = print_number(c, kMul + 1);
            }
            for (const auto& [b, ex] : e->factors()) {
                std::string piece = ex->is_one()
                                        ? print(b, kMul + 1)
                                        : print(b, kPow + 1) + "^" + print(ex, kPow + 1);
                if (!out.empty()) out += "*";
                out += piece;
            }
            if (negate) out = "-" + out;
            return wrap(std::move(out), parent_prec > kMul || (negate && parent_prec > kAdd));
        }
        case ExprKind::Pow:
            return wrap(print(e->base(), kPow + 1) + "^" + print(e->exp(), kPow + 1),
                        parent_prec > kPow);
        case ExprKind::Func: {
            const auto& def = FuncRegistry::instance().get(e->func_id());
            std::string out = def.name + "(";
            for (size_t i = 0; i < e->args().size(); ++i) {
                if (i) out += ", ";
                out += print(e->args()[i], 0);
            }
            return out + ")";
        }
    }
    return "?";
}

}  // namespace

std::string to_string(const ExprPtr& e) { return print(e, 0); }

}  // namespace frontier
