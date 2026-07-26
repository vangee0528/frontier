#include "frontier/number.hpp"

#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>

#include "frontier/checked_int.hpp"
#include "frontier/portable.hpp"
#include "frontier/error.hpp"

namespace frontier {

namespace {

size_t hash_combine(size_t seed, size_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

int64_t gcd64(int64_t a, int64_t b) {
    if (a < 0) a = (a == INT64_MIN) ? INT64_MIN : -a;  // MIN 的 gcd 仍正确终止
    if (b < 0) b = (b == INT64_MIN) ? INT64_MIN : -b;
    while (a != 0) {
        const int64_t t = a;
        a = b % a;
        b = t;
    }
    return b;
}

}  // namespace

Number Number::integer(int64_t v) {
    Number n;
    n.kind_ = Kind::Int;
    n.num_ = v;
    return n;
}

Number Number::real(double v) {
    Number n;
    n.kind_ = Kind::Real;
    n.real_ = v;
    return n;
}

Number Number::rational(int64_t num, int64_t den) {
    if (den == 0) throw DomainError("rational with zero denominator");
    // INT64_MIN 取绝对值会溢出，直接走浮点降级（极端罕见）
    if (num == INT64_MIN || den == INT64_MIN)
        return real(static_cast<double>(num) / static_cast<double>(den));
    if (den < 0) {
        num = -num;
        den = -den;
    }
    const int64_t g = std::gcd(num < 0 ? -num : num, den);
    if (g > 1) {
        num /= g;
        den /= g;
    }
    if (den == 1) return integer(num);
    Number n;
    n.kind_ = Kind::Rational;
    n.num_ = num;
    n.den_ = den;
    return n;
}

bool Number::is_zero() const {
    switch (kind_) {
        case Kind::Int: return num_ == 0;
        case Kind::Rational: return false;  // 最简形式下分子非零
        case Kind::Real: return real_ == 0.0;
    }
    return false;
}

bool Number::is_one() const {
    switch (kind_) {
        case Kind::Int: return num_ == 1;
        case Kind::Rational: return false;
        case Kind::Real: return real_ == 1.0;
    }
    return false;
}

bool Number::is_minus_one() const {
    switch (kind_) {
        case Kind::Int: return num_ == -1;
        case Kind::Rational: return false;
        case Kind::Real: return real_ == -1.0;
    }
    return false;
}

bool Number::is_negative() const {
    switch (kind_) {
        case Kind::Int:
        case Kind::Rational: return num_ < 0;
        case Kind::Real: return real_ < 0.0;
    }
    return false;
}

int64_t Number::int_value() const {
    assert(kind_ == Kind::Int);
    return num_;
}

double Number::to_double() const {
    switch (kind_) {
        case Kind::Int: return static_cast<double>(num_);
        case Kind::Rational:
            return static_cast<double>(num_) / static_cast<double>(den_);
        case Kind::Real: return real_;
    }
    return 0.0;
}

Number Number::operator-() const {
    switch (kind_) {
        case Kind::Int:
            if (num_ == INT64_MIN) return real(-static_cast<double>(num_));
            return integer(-num_);
        case Kind::Rational: return rational(-num_, den_);
        case Kind::Real: return real(-real_);
    }
    return {};
}

Number Number::operator+(const Number& o) const {
    if (kind_ == Kind::Real || o.kind_ == Kind::Real)
        return real(to_double() + o.to_double());
    // a/b + c/d，先用 g=gcd(b,d) 预约分（Knuth 4.5.1），每步 checked，
    // 溢出降级 Real
    const int64_t a = num_, b = den_, c = o.num_, d = o.den_;
    const int64_t g = gcd64(b, d);
    const int64_t b1 = b / g, d1 = d / g;
    int64_t ad, cb, num, den;
    if (checked::mul(a, d1, &ad) || checked::mul(c, b1, &cb) ||
        checked::add(ad, cb, &num) || checked::mul(b1, d, &den))
        return real(to_double() + o.to_double());
    return rational(num, den);
}

Number Number::operator-(const Number& o) const { return *this + (-o); }

Number Number::operator*(const Number& o) const {
    if (kind_ == Kind::Real || o.kind_ == Kind::Real)
        return real(to_double() * o.to_double());
    // a/b · c/d，交叉预约分：g1=gcd(a,d)，g2=gcd(c,b)
    const int64_t g1 = gcd64(num_, o.den_);
    const int64_t g2 = gcd64(o.num_, den_);
    const int64_t a = (g1 > 1) ? num_ / g1 : num_;
    const int64_t d = (g1 > 1) ? o.den_ / g1 : o.den_;
    const int64_t c = (g2 > 1) ? o.num_ / g2 : o.num_;
    const int64_t b = (g2 > 1) ? den_ / g2 : den_;
    int64_t num, den;
    if (checked::mul(a, c, &num) || checked::mul(b, d, &den))
        return real(to_double() * o.to_double());
    return rational(num, den);
}

Number Number::operator/(const Number& o) const {
    if (kind_ == Kind::Real || o.kind_ == Kind::Real)
        return real(to_double() / o.to_double());
    if (o.num_ == 0) throw DomainError("division by zero");
    if (o.num_ == INT64_MIN)  // 倒数分母无法取正，走浮点
        return real(to_double() / o.to_double());
    return *this * rational(o.den_, o.num_);
}

Number Number::inverse() const { return integer(1) / *this; }

std::optional<Number> Number::pow(const Number& base, const Number& exp) {
    // 整数指数：精确快速幂（含负指数 → 倒数），溢出降级 Real
    if (exp.kind_ == Kind::Int) {
        int64_t e = exp.num_;
        if (e == 0) return integer(1);
        if (base.kind_ == Kind::Real)
            return real(std::pow(base.real_, static_cast<double>(e)));
        const bool neg_exp = e < 0;
        if (neg_exp) {
            if (base.num_ == 0) throw DomainError("zero to a negative power");
            e = -e;
        }
        const double dfall =
            std::pow(base.to_double(), static_cast<double>(neg_exp ? -e : e));
        if (e > 512) return real(dfall);  // 防爆炸：巨大指数直接走浮点
        int64_t rn = 1, rd = 1, bn = base.num_, bd = base.den_;
        bool overflow = false;
        for (int64_t k = e; k > 0 && !overflow; k >>= 1) {
            if (k & 1)
                overflow = checked::mul(rn, bn, &rn) || checked::mul(rd, bd, &rd);
            if (k > 1 && !overflow)
                overflow = checked::mul(bn, bn, &bn) || checked::mul(bd, bd, &bd);
        }
        if (overflow) return real(dfall);
        if (neg_exp) {
            std::swap(rn, rd);
            if (rd == 0) throw DomainError("zero to a negative power");
        }
        return rational(rn, rd);
    }

    const double b = base.to_double();
    const double x = exp.to_double();

    // 负底数配非整数指数：复数域，交由调用方保持符号形式
    if (b < 0.0) return std::nullopt;

    // 正整数底数配有理指数：尝试精确整数根（如 4^(1/2) = 2）
    if (base.kind_ == Kind::Int && exp.kind_ == Kind::Rational && base.num_ > 0) {
        const int64_t p = exp.num_, q = exp.den_;
        const double root = std::pow(static_cast<double>(base.num_), 1.0 / q);
        const int64_t r = std::llround(root);
        if (r > 0) {
            int64_t acc = 1;
            bool ok = true;
            for (int64_t k = 0; k < q && ok; ++k)
                ok = !checked::mul(acc, r, &acc);
            if (ok && acc == base.num_) {
                auto exact = pow(integer(r), integer(p));
                if (exact) return exact;
            }
        }
    }

    return real(std::pow(b, x));
}

bool Number::operator==(const Number& o) const {
    if (kind_ != o.kind_) return false;
    switch (kind_) {
        case Kind::Int: return num_ == o.num_;
        case Kind::Rational: return num_ == o.num_ && den_ == o.den_;
        case Kind::Real:
            // 位级比较：NaN == NaN 为真，保证 hash-consing 一致性
            return portable::f64_bits(real_) == portable::f64_bits(o.real_);
    }
    return false;
}

int Number::compare(const Number& a, const Number& b) {
    // 精确 vs 精确：交叉相乘避免浮点误差；乘积溢出退化为 double 比较
    //（此时数值必然巨大，double 53 位精度足以区分或判平）
    if (a.is_exact() && b.is_exact()) {
        int64_t lhs, rhs;
        if (checked::mul(a.num_, b.den_, &lhs) ||
            checked::mul(b.num_, a.den_, &rhs)) {
            const double x = a.to_double(), y = b.to_double();
            if (x < y) return -1;
            if (x > y) return 1;
        } else if (lhs != rhs) {
            return lhs < rhs ? -1 : 1;
        }
    } else {
        const double x = a.to_double(), y = b.to_double();
        if (x < y) return -1;
        if (x > y) return 1;
        if (std::isnan(x) != std::isnan(y)) return std::isnan(x) ? 1 : -1;
    }
    // 数值相等：按 Kind 打破平局（Int < Rational < Real）
    const auto ka = static_cast<int>(a.kind_), kb = static_cast<int>(b.kind_);
    if (ka != kb) return ka < kb ? -1 : 1;
    return 0;
}

size_t Number::hash() const {
    size_t seed = static_cast<size_t>(kind_);
    switch (kind_) {
        case Kind::Int:
            seed = hash_combine(seed, std::hash<int64_t>{}(num_));
            break;
        case Kind::Rational:
            seed = hash_combine(seed, std::hash<int64_t>{}(num_));
            seed = hash_combine(seed, std::hash<int64_t>{}(den_));
            break;
        case Kind::Real:
            seed = hash_combine(seed, std::hash<uint64_t>{}(portable::f64_bits(real_)));
            break;
    }
    return seed;
}

std::string Number::to_string() const {
    switch (kind_) {
        case Kind::Int: return std::to_string(num_);
        case Kind::Rational:
            return std::to_string(num_) + "/" + std::to_string(den_);
        case Kind::Real: {
            // 往返安全表示（如 0.5 而非 0.500000）
            return portable::format_double(real_);
        }
    }
    return {};
}

}  // namespace frontier
