#include "doctest.h"
#include "frontier/error.hpp"
#include "frontier/number.hpp"

using frontier::DomainError;
using frontier::Number;

TEST_CASE("rational 规范化") {
    auto r = Number::rational(4, 8);
    CHECK(r.kind() == Number::Kind::Rational);
    CHECK(r.num() == 1);
    CHECK(r.den() == 2);

    CHECK(Number::rational(6, 3) == Number::integer(2));   // den==1 退化为 Int
    CHECK(Number::rational(1, -2) == Number::rational(-1, 2));  // 分母恒正
    CHECK_THROWS_AS(Number::rational(1, 0), DomainError);
}

TEST_CASE("精确算术") {
    auto half = Number::rational(1, 2);
    auto third = Number::rational(1, 3);
    CHECK(half + third == Number::rational(5, 6));
    CHECK(half * third == Number::rational(1, 6));
    CHECK(half - half == Number::integer(0));
    CHECK(half / third == Number::rational(3, 2));
    CHECK((Number::integer(2) / Number::integer(4)) == half);
    CHECK_THROWS_AS(Number::integer(1) / Number::integer(0), DomainError);
}

TEST_CASE("Real 传染性") {
    auto x = Number::integer(1) + Number::real(0.5);
    CHECK(x.kind() == Number::Kind::Real);
    CHECK(x.to_double() == doctest::Approx(1.5));
}

TEST_CASE("溢出降级为 Real") {
    auto big = Number::integer(INT64_MAX);
    auto r = big * big;
    CHECK(r.kind() == Number::Kind::Real);
}

TEST_CASE("幂运算") {
    // 精确整数幂
    auto r = Number::pow(Number::integer(2), Number::integer(10));
    REQUIRE(r.has_value());
    CHECK(*r == Number::integer(1024));

    // 负指数 → 有理数
    r = Number::pow(Number::integer(2), Number::integer(-2));
    REQUIRE(r.has_value());
    CHECK(*r == Number::rational(1, 4));

    // 精确整数根：4^(1/2) = 2
    r = Number::pow(Number::integer(4), Number::rational(1, 2));
    REQUIRE(r.has_value());
    CHECK(*r == Number::integer(2));

    // 无精确根：2^(1/2) → Real（表达式层负责保持符号）
    r = Number::pow(Number::integer(2), Number::rational(1, 2));
    REQUIRE(r.has_value());
    CHECK(r->kind() == Number::Kind::Real);

    // 负底数非整指数 → nullopt（复数域）
    r = Number::pow(Number::integer(-2), Number::rational(1, 2));
    CHECK(!r.has_value());

    // 有理底数整数幂
    r = Number::pow(Number::rational(2, 3), Number::integer(2));
    REQUIRE(r.has_value());
    CHECK(*r == Number::rational(4, 9));
}

TEST_CASE("全序与哈希") {
    CHECK(Number::compare(Number::integer(1), Number::integer(2)) < 0);
    CHECK(Number::compare(Number::rational(1, 2), Number::integer(1)) < 0);
    // 数值相等但 Kind 不同：Int < Real
    CHECK(Number::compare(Number::integer(1), Number::real(1.0)) < 0);
    CHECK(Number::compare(Number::integer(1), Number::integer(1)) == 0);

    CHECK(Number::integer(3).hash() == Number::integer(3).hash());
    CHECK(Number::rational(1, 2).hash() == Number::rational(2, 4).hash());
}

TEST_CASE("打印") {
    CHECK(Number::integer(-5).to_string() == "-5");
    CHECK(Number::rational(3, 4).to_string() == "3/4");
    CHECK(Number::real(0.5).to_string() == "0.5");
}
