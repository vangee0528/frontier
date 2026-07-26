#include "doctest.h"
#include "frontier/builders.hpp"
#include "frontier/diff.hpp"
#include "frontier/error.hpp"
#include "frontier/printer.hpp"

using namespace frontier;

// interning 让「求导结果正确」可以用指针相等断言——比字符串快照更严格
TEST_CASE("基本求导规则") {
    auto x = symbol("x");
    auto y = symbol("y");

    CHECK(diff(x, x)->is_one());
    CHECK(diff(y, x)->is_zero());
    CHECK(diff(integer(42), x)->is_zero());

    // d/dx x^3 = 3x^2
    CHECK(diff(pow(x, integer(3)), x).get() == mul(integer(3), pow(x, integer(2))).get());

    // d/dx (2x + 3y) = 2
    auto e = add(mul(integer(2), x), mul(integer(3), y));
    CHECK(diff(e, x)->number() == Number::integer(2));
}

TEST_CASE("函数与链式法则") {
    auto x = symbol("x");
    auto y = symbol("y");

    // d/dx sin(x) = cos(x)
    CHECK(diff(func("sin", {x}), x).get() == func("cos", {x}).get());

    // d/dx sin(x*y) = y*cos(x*y)
    auto sxy = func("sin", {mul(x, y)});
    CHECK(diff(sxy, x).get() == mul(y, func("cos", {mul(x, y)})).get());

    // d/dx exp(-x^2) = -2x·exp(-x^2)
    auto g = func("exp", {neg(pow(x, integer(2)))});
    CHECK(diff(g, x).get() == mul({integer(-2), x, g}).get());

    // d/dx log(x) = 1/x
    CHECK(diff(func("log", {x}), x).get() == pow(x, minus_one()).get());
}

TEST_CASE("乘积与商法则") {
    auto x = symbol("x");

    // d/dx (x·sin x) = sin x + x·cos x
    auto e = mul(x, func("sin", {x}));
    auto expected = add(func("sin", {x}), mul(x, func("cos", {x})));
    CHECK(diff(e, x).get() == expected.get());

    // d/dx (1/x) = -x^(-2)
    CHECK(diff(pow(x, minus_one()), x).get() == neg(pow(x, integer(-2))).get());
}

TEST_CASE("一般幂：x^x") {
    auto x = symbol("x");
    auto e = pow(x, x);
    // x^x·(log x + 1)
    auto expected = mul(e, add(func("log", {x}), one()));
    CHECK(diff(e, x).get() == expected.get());
}

TEST_CASE("对非 Symbol 求导报错") {
    auto x = symbol("x");
    CHECK_THROWS_AS(diff(x, add(x, one())), Error);
}

TEST_CASE("DAG 共享子表达式") {
    auto x = symbol("x");
    // u = sin(x)，f = u + u^2；f' = cos x + 2·sin x·cos x
    auto u = func("sin", {x});
    auto f = add(u, pow(u, integer(2)));
    auto expected = add(func("cos", {x}),
                        mul({integer(2), u, func("cos", {x})}));
    CHECK(diff(f, x).get() == expected.get());
}
