#include "doctest.h"
#include "frontier/builders.hpp"
#include "frontier/error.hpp"
#include "frontier/tape.hpp"

using namespace frontier;

namespace {
size_t count_kind(const Tape& t, TapeOp::Kind k) {
    size_t n = 0;
    for (const auto& op : t.ops)
        if (op.kind == k) ++n;
    return n;
}
}  // namespace

TEST_CASE("CSE：共享子表达式只发射一次") {
    auto x = symbol("x");
    auto y = symbol("y");
    auto xy = mul(x, y);
    const ExprPtr exprs[] = {func("sin", {xy}), func("cos", {xy})};
    const ExprPtr args[] = {x, y};

    Tape t = lower(exprs, args);
    // 2 输入 + 1 乘法 + 2 调用 = 5 条指令
    CHECK(t.ops.size() == 5);
    CHECK(t.outputs.size() == 2);
    CHECK(count_kind(t, TapeOp::Kind::Mul) == 1);
    CHECK(count_kind(t, TapeOp::Kind::Call) == 2);
}

TEST_CASE("常量去重") {
    auto x = symbol("x");
    const ExprPtr exprs[] = {add(x, integer(2)), mul(integer(2), x)};
    const ExprPtr args[] = {x};

    Tape t = lower(exprs, args);
    CHECK(count_kind(t, TapeOp::Kind::Const) == 1);
}

TEST_CASE("自由变量缺失报错") {
    auto x = symbol("x");
    auto y = symbol("y");
    const ExprPtr exprs[] = {add(x, y)};
    const ExprPtr args[] = {x};
    CHECK_THROWS_AS(lower(exprs, args), CompileError);
}

TEST_CASE("非 Symbol 实参报错") {
    auto x = symbol("x");
    const ExprPtr exprs[] = {x};
    const ExprPtr args[] = {add(x, one())};
    CHECK_THROWS_AS(lower(exprs, args), CompileError);
}

TEST_CASE("输出编号与结构") {
    auto x = symbol("x");
    // f = x：输出直接指向 Input 指令
    const ExprPtr exprs[] = {x};
    const ExprPtr args[] = {x};
    Tape t = lower(exprs, args);
    CHECK(t.ops.size() == 1);
    CHECK(t.outputs[0] == 0);
    CHECK(t.ops[0].kind == TapeOp::Kind::Input);

    // 纯常量输出
    const ExprPtr exprs2[] = {integer(7)};
    Tape t2 = lower(exprs2, args);
    CHECK(t2.outputs.size() == 1);
    CHECK(t2.ops[t2.outputs[0]].kind == TapeOp::Kind::Const);
}
