#include "frontier/func_registry.hpp"

#include <cmath>

#include "frontier/builders.hpp"
#include "frontier/error.hpp"

namespace frontier {

namespace {

using Args = std::vector<ExprPtr>;

double eval_sign(const double* a) {
    return a[0] > 0.0 ? 1.0 : (a[0] < 0.0 ? -1.0 : 0.0);
}

CodegenSpec intrin(const char* s) {
    return {CodegenSpec::Lower::Intrinsic, s};
}
CodegenSpec libm(const char* s) {
    return {CodegenSpec::Lower::LibmCall, s};
}

}  // namespace

FuncRegistry& FuncRegistry::instance() {
    static FuncRegistry* r = new FuncRegistry();  // 刻意泄漏，同 intern 表
    return *r;
}

FuncId FuncRegistry::add(FuncDef def) {
    if (find(def.name)) throw Error("function already registered: " + def.name);
    if (defs_.size() >= UINT16_MAX) throw Error("function registry full");
    names_.push_back(def.name);
    defs_.push_back(std::move(def));
    return static_cast<FuncId>(defs_.size() - 1);
}

const FuncDef& FuncRegistry::get(FuncId id) const {
    if (id >= defs_.size()) throw Error("invalid FuncId");
    return defs_[id];
}

std::optional<FuncId> FuncRegistry::find(std::string_view name) const {
    for (size_t i = 0; i < names_.size(); ++i)
        if (names_[i] == name) return static_cast<FuncId>(i);
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// 内置函数。导数规则返回对 args[0] 的偏导（一元函数）。
// ---------------------------------------------------------------------------

FuncRegistry::FuncRegistry() {
    auto u = [this](const char* name,
                    std::function<ExprPtr(const Args&, int)> deriv,
                    double (*eval)(const double*), CodegenSpec cg) {
        names_.emplace_back(name);
        defs_.push_back(FuncDef{name, 1, std::move(deriv), eval, std::move(cg)});
    };

    u("sin", [](const Args& a, int) { return func("cos", {a[0]}); },
      +[](const double* a) { return std::sin(a[0]); }, intrin("llvm.sin.f64"));

    u("cos", [](const Args& a, int) { return neg(func("sin", {a[0]})); },
      +[](const double* a) { return std::cos(a[0]); }, intrin("llvm.cos.f64"));

    u("tan",  // 1 + tan²x
      [](const Args& a, int) { return frontier::add(one(), pow(func("tan", {a[0]}), integer(2))); },
      +[](const double* a) { return std::tan(a[0]); }, libm("tan"));

    u("asin",  // (1-x²)^(-1/2)
      [](const Args& a, int) {
          return pow(sub(one(), pow(a[0], integer(2))), rational(-1, 2));
      },
      +[](const double* a) { return std::asin(a[0]); }, libm("asin"));

    u("acos",
      [](const Args& a, int) {
          return neg(pow(sub(one(), pow(a[0], integer(2))), rational(-1, 2)));
      },
      +[](const double* a) { return std::acos(a[0]); }, libm("acos"));

    u("atan",  // (1+x²)^(-1)
      [](const Args& a, int) { return pow(frontier::add(one(), pow(a[0], integer(2))), minus_one()); },
      +[](const double* a) { return std::atan(a[0]); }, libm("atan"));

    u("sinh", [](const Args& a, int) { return func("cosh", {a[0]}); },
      +[](const double* a) { return std::sinh(a[0]); }, libm("sinh"));

    u("cosh", [](const Args& a, int) { return func("sinh", {a[0]}); },
      +[](const double* a) { return std::cosh(a[0]); }, libm("cosh"));

    u("tanh",  // 1 - tanh²x
      [](const Args& a, int) { return sub(one(), pow(func("tanh", {a[0]}), integer(2))); },
      +[](const double* a) { return std::tanh(a[0]); }, libm("tanh"));

    u("exp", [](const Args& a, int) { return func("exp", {a[0]}); },
      +[](const double* a) { return std::exp(a[0]); }, intrin("llvm.exp.f64"));

    u("log",  // 1/x
      [](const Args& a, int) { return pow(a[0], minus_one()); },
      +[](const double* a) { return std::log(a[0]); }, intrin("llvm.log.f64"));

    // sqrt 经 rewrite 规范化为 x^(1/2)：保证与其他幂合并
    // （sqrt(x)·x^(-1/2) = 1）；数值端由后端将 Pow(x, 0.5) 特化为 sqrt 调用
    names_.emplace_back("sqrt");
    defs_.push_back(FuncDef{
        "sqrt", 1,
        [](const Args& a, int) { return mul(rational(1, 2), pow(a[0], rational(-1, 2))); },
        +[](const double* a) { return std::sqrt(a[0]); },
        intrin("llvm.sqrt.f64"),
        [](const Args& a) { return pow(a[0], rational(1, 2)); }});

    u("abs", [](const Args& a, int) { return func("sign", {a[0]}); },
      +[](const double* a) { return std::fabs(a[0]); }, intrin("llvm.fabs.f64"));

    u("sign",  // 几乎处处为 0
      [](const Args&, int) { return zero(); },
      eval_sign, {CodegenSpec::Lower::Custom, "sign"});

    // max/min：次梯度导数 d max(a,b)/da = (1+sign(a-b))/2；
    // codegen 走 llvm.maxnum/minnum（NaN 语义同 fmax/fmin）
    auto minmax_deriv = [](bool is_max) {
        return [is_max](const Args& a, int i) -> ExprPtr {
            ExprPtr s = func("sign", {sub(a[0], a[1])});
            if ((i == 1) == is_max) s = neg(s);  // max 对 b / min 对 a 取负
            return mul(rational(1, 2), frontier::add(one(), s));
        };
    };
    names_.emplace_back("max");
    defs_.push_back(FuncDef{
        "max", 2, minmax_deriv(true),
        +[](const double* a) { return a[0] > a[1] ? a[0] : a[1]; },
        intrin("llvm.maxnum.f64")});
    names_.emplace_back("min");
    defs_.push_back(FuncDef{
        "min", 2, minmax_deriv(false),
        +[](const double* a) { return a[0] < a[1] ? a[0] : a[1]; },
        intrin("llvm.minnum.f64")});

    u("asinh",  // (x²+1)^(-1/2)
      [](const Args& a, int) {
          return pow(frontier::add(pow(a[0], integer(2)), one()), rational(-1, 2));
      },
      +[](const double* a) { return std::asinh(a[0]); }, libm("asinh"));

    u("acosh",  // (x²-1)^(-1/2)
      [](const Args& a, int) {
          return pow(sub(pow(a[0], integer(2)), one()), rational(-1, 2));
      },
      +[](const double* a) { return std::acosh(a[0]); }, libm("acosh"));

    u("atanh",  // (1-x²)^(-1)
      [](const Args& a, int) {
          return pow(sub(one(), pow(a[0], integer(2))), minus_one());
      },
      +[](const double* a) { return std::atanh(a[0]); }, libm("atanh"));

    u("floor", [](const Args&, int) { return zero(); },  // 几乎处处 0
      +[](const double* a) { return std::floor(a[0]); }, intrin("llvm.floor.f64"));

    u("ceil", [](const Args&, int) { return zero(); },
      +[](const double* a) { return std::ceil(a[0]); }, intrin("llvm.ceil.f64"));

    // 比较谓词：返回 0.0/1.0 指示值；导数几乎处处为 0。
    // codegen 为 Custom：fcmp + select（可向量化，无分支）
    auto cmp = [this](const char* name, double (*eval)(const double*)) {
        names_.emplace_back(name);
        defs_.push_back(FuncDef{
            name, 2, [](const Args&, int) { return zero(); }, eval,
            {CodegenSpec::Lower::Custom, name}});
    };
    cmp("lt", +[](const double* a) { return a[0] < a[1] ? 1.0 : 0.0; });
    cmp("le", +[](const double* a) { return a[0] <= a[1] ? 1.0 : 0.0; });
    cmp("gt", +[](const double* a) { return a[0] > a[1] ? 1.0 : 0.0; });
    cmp("ge", +[](const double* a) { return a[0] >= a[1] ? 1.0 : 0.0; });

    // where(cond, a, b)：cond≠0 取 a 否则 b。分支导数：
    // ∂/∂a = where(cond,1,0)，∂/∂b = where(cond,0,1)，∂/∂cond = 0
    names_.emplace_back("where");
    defs_.push_back(FuncDef{
        "where", 3,
        [](const Args& a, int i) -> ExprPtr {
            if (i == 0) return zero();
            return func("where", {a[0], i == 1 ? one() : zero(),
                                  i == 1 ? zero() : one()});
        },
        +[](const double* a) { return a[0] != 0.0 ? a[1] : a[2]; },
        {CodegenSpec::Lower::Custom, "where"}});

    // erf/erfc：d erf/dx = 2/√π·exp(-x²)
    const double two_over_sqrt_pi = 1.1283791670955126;
    auto erf_deriv = [two_over_sqrt_pi](bool negate) {
        return [two_over_sqrt_pi, negate](const Args& a, int) -> ExprPtr {
            ExprPtr d = mul(real(two_over_sqrt_pi),
                            func("exp", {neg(pow(a[0], integer(2)))}));
            return negate ? neg(d) : d;
        };
    };
    names_.emplace_back("erf");
    defs_.push_back(FuncDef{
        "erf", 1, erf_deriv(false),
        +[](const double* a) { return std::erf(a[0]); }, libm("erf")});
    names_.emplace_back("erfc");
    defs_.push_back(FuncDef{
        "erfc", 1, erf_deriv(true),
        +[](const double* a) { return std::erfc(a[0]); }, libm("erfc")});

    // atan2(y, x)：∂/∂y = x/(x²+y²)，∂/∂x = -y/(x²+y²)
    names_.emplace_back("atan2");
    defs_.push_back(FuncDef{
        "atan2", 2,
        [](const Args& a, int i) {
            const ExprPtr denom = frontier::add(pow(a[0], integer(2)), pow(a[1], integer(2)));
            const ExprPtr num = (i == 0) ? a[1] : neg(a[0]);
            return mul(num, pow(denom, minus_one()));
        },
        +[](const double* a) { return std::atan2(a[0], a[1]); },
        libm("atan2")});
}

}  // namespace frontier
