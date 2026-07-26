#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace frontier {

// 精确数值塔：Int ⊂ Rational ⊂ Real(double)。
//
// 不变量：
//  - Rational 恒为最简形式，den > 0 且 den != 1（den==1 立即退化为 Int）；
//  - Int/Rational 运算在 int64 溢出时降级为 Real（经 __int128 检测）；
//  - 只有 Real 参与的运算才引入浮点误差（架构不变量 #3）。
//
// 扩展点：新增数值类型（大整数、复数）只改本文件与 number.cpp，
// 表达式层通过 Number 的运算接口与之隔离。
class Number {
public:
    enum class Kind : uint8_t { Int, Rational, Real };

    Number() : kind_(Kind::Int), num_(0) {}

    static Number integer(int64_t v);
    static Number rational(int64_t num, int64_t den);  // den==0 抛 DomainError
    static Number real(double v);

    Kind kind() const { return kind_; }
    bool is_exact() const { return kind_ != Kind::Real; }
    bool is_int() const { return kind_ == Kind::Int; }
    bool is_zero() const;
    bool is_one() const;
    bool is_minus_one() const;
    bool is_negative() const;

    int64_t int_value() const;  // 仅 Kind::Int 合法
    int64_t num() const { return num_; }
    int64_t den() const { return den_; }
    double to_double() const;

    Number operator-() const;
    Number operator+(const Number&) const;
    Number operator-(const Number&) const;
    Number operator*(const Number&) const;
    Number operator/(const Number&) const;  // 精确路径除零抛 DomainError
    Number inverse() const;

    // 幂运算：结果可精确表示则返回精确值，否则 Real。
    // 负底数配非整数指数（复数域）返回 nullopt，调用方保持符号形式不求值。
    static std::optional<Number> pow(const Number& base, const Number& exp);

    bool operator==(const Number&) const;
    bool operator!=(const Number& o) const { return !(*this == o); }

    // 全序：先按数值大小，数值相等时按 Kind 打破平局。用于表达式规范排序。
    static int compare(const Number& a, const Number& b);

    size_t hash() const;
    std::string to_string() const;

private:
    Kind kind_;
    int64_t num_ = 0;  // Int: 值；Rational: 分子
    int64_t den_ = 1;  // Rational: 分母（>0）
    double real_ = 0.0;
};

}  // namespace frontier
