#include "doctest.h"
#include "frontier/builders.hpp"
#include "frontier/error.hpp"
#include "frontier/printer.hpp"

using namespace frontier;

TEST_CASE("hash-consing：指针相等 ⇔ 结构相等") {
    auto x = symbol("x");
    auto y = symbol("y");
    CHECK(symbol("x").get() == x.get());

    // 交换律经规范排序归一
    CHECK(add(x, y).get() == add(y, x).get());
    CHECK(mul(x, y).get() == mul(y, x).get());

    // 结合律经扁平化归一：(x+y)+1 == x+(y+1)
    CHECK(add(add(x, y), one()).get() == add(x, add(y, one())).get());
}

TEST_CASE("同类项合并") {
    auto x = symbol("x");
    auto y = symbol("y");

    // 2x + 3x = 5x
    auto e = add(mul(integer(2), x), mul(integer(3), x));
    CHECK(to_string(e) == "5*x");

    // x - x = 0
    CHECK(sub(x, x)->is_zero());

    // x + y - y = x
    CHECK(add(add(x, y), neg(y)).get() == x.get());
}

TEST_CASE("同底数幂合并") {
    auto x = symbol("x");

    // x * x = x^2
    auto e = mul(x, x);
    CHECK(e->kind() == ExprKind::Pow);
    CHECK(to_string(e) == "x^2");

    // x^2 * x^3 = x^5
    CHECK(to_string(mul(pow(x, integer(2)), pow(x, integer(3)))) == "x^5");

    // x / x = 1
    CHECK(div(x, x)->is_one());
}

TEST_CASE("常量折叠与单位元") {
    auto x = symbol("x");

    CHECK(add(integer(2), integer(3))->number() == Number::integer(5));
    CHECK(mul(integer(2), rational(1, 2))->is_one());

    CHECK(add(x, zero()).get() == x.get());   // x + 0 = x
    CHECK(mul(x, one()).get() == x.get());    // x * 1 = x
    CHECK(mul(x, zero())->is_zero());         // x * 0 = 0
    CHECK(pow(x, one()).get() == x.get());    // x^1 = x
    CHECK(pow(x, zero())->is_one());          // x^0 = 1
}

TEST_CASE("数值系数对 Add 的分配律") {
    auto x = symbol("x");
    auto y = symbol("y");

    // 2*(x+y) = 2x + 2y（利于同类项合并）
    auto e = mul(integer(2), add(x, y));
    CHECK(e->kind() == ExprKind::Add);
    CHECK(to_string(e) == "2*x + 2*y");

    // 2*(x+y) - 2x = 2y
    CHECK(to_string(sub(e, mul(integer(2), x))) == "2*y");
}

TEST_CASE("Pow 规范化") {
    auto x = symbol("x");
    auto y = symbol("y");

    // (x^2)^3 = x^6（整数指数）
    CHECK(to_string(pow(pow(x, integer(2)), integer(3))) == "x^6");

    // (x*y)^2 = x^2*y^2（整数指数分配）
    CHECK(to_string(pow(mul(x, y), integer(2))) == "x^2*y^2");

    // (x^y)^z 不合并（非整数指数不安全）
    auto z = symbol("z");
    auto e = pow(pow(x, y), z);
    CHECK(e->kind() == ExprKind::Pow);
    CHECK(e->base()->kind() == ExprKind::Pow);
}

TEST_CASE("精确性规则") {
    // sqrt(2) 保持符号，不折叠成浮点
    auto e = pow(integer(2), rational(1, 2));
    CHECK(e->kind() == ExprKind::Pow);

    // sqrt(4) 精确折叠
    CHECK(pow(integer(4), rational(1, 2))->number() == Number::integer(2));

    // Real 参与则折叠
    CHECK(pow(real(2.0), rational(1, 2))->number().kind() == Number::Kind::Real);

    // sin(0) 保持符号（v1 不做特殊值表）；sin(0.0) 折叠
    CHECK(func("sin", {zero()})->kind() == ExprKind::Func);
    CHECK(func("sin", {real(0.0)})->is_zero());
}

TEST_CASE("函数构造") {
    auto x = symbol("x");
    CHECK(to_string(func("sin", {x})) == "sin(x)");
    CHECK_THROWS_AS(func("nope", {x}), Error);
    CHECK_THROWS_AS(func("sin", {x, x}), Error);
}

TEST_CASE("打印") {
    auto x = symbol("x");
    auto y = symbol("y");

    CHECK(to_string(sub(x, y)) == "x - y");
    CHECK(to_string(add(mul(integer(2), x), integer(1))) == "2*x + 1");
    CHECK(to_string(neg(x)) == "-x");
    CHECK(to_string(pow(add(x, y), integer(2))) == "(x + y)^2");
    CHECK(to_string(mul(x, pow(y, integer(-1)))) == "x*y^(-1)");
    CHECK(to_string(pow(x, rational(-1, 2))) == "x^(-1/2)");
}

TEST_CASE("除法即负幂") {
    auto x = symbol("x");
    auto y = symbol("y");
    auto e = div(x, y);
    CHECK(e->kind() == ExprKind::Mul);
    CHECK(to_string(e) == "x*y^(-1)");
}

TEST_CASE("同底合并出整数指数时回流 pow 展开（fuzz 回归）") {
    auto x = symbol("x");
    auto nx = neg(x);  // Mul(-1, x)

    // (-x)^(1/2) · (-x)^(-3/2) = (-x)^(-1) = -x^(-1)（符号必须保留）
    auto e = mul(pow(nx, rational(1, 2)), pow(nx, rational(-3, 2)));
    CHECK(e.get() == neg(pow(x, integer(-1))).get());

    // (x^y)^(1/3) · (x^y)^(2/3) = x^y（Pow 底整数指数同样回流）
    auto y = symbol("y");
    auto p = pow(x, y);
    CHECK(mul(pow(p, rational(1, 3)), pow(p, rational(2, 3))).get() == p.get());
}
