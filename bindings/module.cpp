// pybind11 薄绑定：只做类型转换与异常映射，不含任何算法逻辑
// （见 docs/internals.md 不变量一节）。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "frontier/builders.hpp"
#include "frontier/codegen/llvm_text.hpp"
#include "frontier/diff.hpp"
#include "frontier/error.hpp"
#include "frontier/func_registry.hpp"
#include "frontier/printer.hpp"
#include "frontier/subs.hpp"
#include "frontier/tape.hpp"
#include "frontier/visitor.hpp"

namespace py = pybind11;
using namespace frontier;

namespace {

// Python 侧句柄：包一层以规避 shared_ptr<const T> holder 的限制
struct PyExpr {
    ExprPtr p;
};

// ---------------------------------------------------------------------------
// pickle 状态：扁平后序节点表（保持 DAG 共享，线性大小，无递归深度限制）。
// Number 编码 ('i',v)/('q',p,q)/('r',double)；节点编码见 encode/decode。
// ---------------------------------------------------------------------------
py::tuple encode_number(const Number& n) {
    switch (n.kind()) {
        case Number::Kind::Int: return py::make_tuple("i", n.int_value());
        case Number::Kind::Rational: return py::make_tuple("q", n.num(), n.den());
        case Number::Kind::Real: return py::make_tuple("r", n.to_double());
    }
    throw Error("unreachable");
}

Number decode_number(const py::tuple& t) {
    const std::string k = py::cast<std::string>(t[0]);
    if (k == "i") return Number::integer(py::cast<int64_t>(t[1]));
    if (k == "q")
        return Number::rational(py::cast<int64_t>(t[1]), py::cast<int64_t>(t[2]));
    if (k == "r") return Number::real(py::cast<double>(t[1]));
    throw Error("bad pickled number tag: " + k);
}

py::list expr_getstate(const ExprPtr& root) {
    // 迭代后序：先子后父，index 表把指针映射为表内下标
    std::vector<const Expr*> order;
    std::unordered_map<const Expr*, int64_t> index;
    {
        std::vector<std::pair<const Expr*, bool>> stack{{root.get(), false}};
        while (!stack.empty()) {
            auto [e, expanded] = stack.back();
            stack.pop_back();
            if (index.count(e)) continue;
            if (expanded) {
                index.emplace(e, static_cast<int64_t>(order.size()));
                order.push_back(e);
                continue;
            }
            stack.push_back({e, true});
            for_each_child(*e, [&](const ExprPtr& c) {
                if (!index.count(c.get())) stack.push_back({c.get(), false});
            });
        }
    }

    py::list nodes;
    for (const Expr* e : order) {
        switch (e->kind()) {
            case ExprKind::Constant:
                nodes.append(py::make_tuple("c", encode_number(e->number())));
                break;
            case ExprKind::Symbol:
                nodes.append(py::make_tuple("s", e->name()));
                break;
            case ExprKind::Add: {
                py::list terms;
                for (const auto& [t, c] : e->terms())
                    terms.append(py::make_tuple(index.at(t.get()), encode_number(c)));
                nodes.append(py::make_tuple("+", encode_number(e->add_coeff()), terms));
                break;
            }
            case ExprKind::Mul: {
                py::list factors;
                for (const auto& [b, x] : e->factors())
                    factors.append(py::make_tuple(index.at(b.get()), index.at(x.get())));
                nodes.append(py::make_tuple("*", encode_number(e->mul_coeff()), factors));
                break;
            }
            case ExprKind::Pow:
                nodes.append(py::make_tuple("^", index.at(e->base().get()),
                                            index.at(e->exp().get())));
                break;
            case ExprKind::Func: {
                py::list fargs;
                for (const auto& a : e->args()) fargs.append(index.at(a.get()));
                nodes.append(py::make_tuple(
                    "f", FuncRegistry::instance().get(e->func_id()).name, fargs));
                break;
            }
        }
    }
    return nodes;
}

ExprPtr expr_setstate(const py::list& nodes) {
    std::vector<ExprPtr> built;
    built.reserve(py::len(nodes));
    for (auto item : nodes) {
        const py::tuple t = py::cast<py::tuple>(item);
        const std::string tag = py::cast<std::string>(t[0]);
        if (tag == "c") {
            built.push_back(constant(decode_number(py::cast<py::tuple>(t[1]))));
        } else if (tag == "s") {
            built.push_back(symbol(py::cast<std::string>(t[1])));
        } else if (tag == "+") {
            std::vector<ExprPtr> parts;
            parts.push_back(constant(decode_number(py::cast<py::tuple>(t[1]))));
            for (auto tc : py::cast<py::list>(t[2])) {
                const py::tuple pair = py::cast<py::tuple>(tc);
                parts.push_back(
                    mul(constant(decode_number(py::cast<py::tuple>(pair[1]))),
                        built.at(py::cast<size_t>(pair[0]))));
            }
            built.push_back(add(std::span<const ExprPtr>(parts)));
        } else if (tag == "*") {
            std::vector<ExprPtr> parts;
            parts.push_back(constant(decode_number(py::cast<py::tuple>(t[1]))));
            for (auto be : py::cast<py::list>(t[2])) {
                const py::tuple pair = py::cast<py::tuple>(be);
                parts.push_back(pow(built.at(py::cast<size_t>(pair[0])),
                                    built.at(py::cast<size_t>(pair[1]))));
            }
            built.push_back(mul(std::span<const ExprPtr>(parts)));
        } else if (tag == "^") {
            built.push_back(pow(built.at(py::cast<size_t>(t[1])),
                                built.at(py::cast<size_t>(t[2]))));
        } else if (tag == "f") {
            std::vector<ExprPtr> fargs;
            for (auto a : py::cast<py::list>(t[2]))
                fargs.push_back(built.at(py::cast<size_t>(a)));
            built.push_back(func(py::cast<std::string>(t[1]), std::move(fargs)));
        } else {
            throw Error("bad pickled node tag: " + tag);
        }
    }
    if (built.empty()) throw Error("empty pickled expression state");
    return built.back();
}

ExprPtr to_expr(py::handle h) {
    if (py::isinstance<PyExpr>(h)) return py::cast<PyExpr&>(h).p;
    if (py::isinstance<py::bool_>(h))
        throw py::type_error("bool cannot be used as a number in expressions");
    if (py::isinstance<py::int_>(h)) {
        try {
            return integer(py::cast<int64_t>(h));
        } catch (const py::cast_error&) {
            throw py::value_error(
                "integer too large for exact int64 arithmetic; pass a float instead");
        }
    }
    if (py::isinstance<py::float_>(h)) return real(py::cast<double>(h));
    throw py::type_error("cannot convert " +
                         std::string(py::str(py::type::of(h))) +
                         " to a frontier expression");
}

PyExpr wrap(ExprPtr e) { return PyExpr{std::move(e)}; }

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Frontier symbolic core (C++)";

    // 异常层级：DomainError/CompileError 在 Error 之后注册（转换器 LIFO，先试子类）
    auto base = py::register_exception<Error>(m, "FrontierError");
    py::register_exception<DomainError>(m, "DomainError", base);
    py::register_exception<CompileError>(m, "CompileError", base);

    py::class_<PyExpr>(m, "Expr")
        .def("__repr__", [](const PyExpr& e) { return to_string(e.p); })
        .def("__str__", [](const PyExpr& e) { return to_string(e.p); })
        .def("__hash__", [](const PyExpr& e) { return static_cast<py::ssize_t>(e.p->hash()); })
        .def("__eq__",
             [](const PyExpr& a, py::handle b) -> py::object {
                 try {
                     // interning：结构相等 ⇔ 指针相等
                     return py::bool_(a.p.get() == to_expr(b).get());
                 } catch (const py::builtin_exception&) {
                     return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
                 }
             })
        .def("__add__", [](const PyExpr& a, py::handle b) { return wrap(add(a.p, to_expr(b))); })
        .def("__radd__", [](const PyExpr& a, py::handle b) { return wrap(add(to_expr(b), a.p)); })
        .def("__sub__", [](const PyExpr& a, py::handle b) { return wrap(sub(a.p, to_expr(b))); })
        .def("__rsub__", [](const PyExpr& a, py::handle b) { return wrap(sub(to_expr(b), a.p)); })
        .def("__mul__", [](const PyExpr& a, py::handle b) { return wrap(mul(a.p, to_expr(b))); })
        .def("__rmul__", [](const PyExpr& a, py::handle b) { return wrap(mul(to_expr(b), a.p)); })
        .def("__truediv__", [](const PyExpr& a, py::handle b) { return wrap(div(a.p, to_expr(b))); })
        .def("__rtruediv__", [](const PyExpr& a, py::handle b) { return wrap(div(to_expr(b), a.p)); })
        .def("__pow__", [](const PyExpr& a, py::handle b) { return wrap(pow(a.p, to_expr(b))); })
        .def("__rpow__", [](const PyExpr& a, py::handle b) { return wrap(pow(to_expr(b), a.p)); })
        .def("__lt__", [](const PyExpr& a, py::handle b) { return wrap(func("lt", {a.p, to_expr(b)})); })
        .def("__le__", [](const PyExpr& a, py::handle b) { return wrap(func("le", {a.p, to_expr(b)})); })
        .def("__gt__", [](const PyExpr& a, py::handle b) { return wrap(func("gt", {a.p, to_expr(b)})); })
        .def("__ge__", [](const PyExpr& a, py::handle b) { return wrap(func("ge", {a.p, to_expr(b)})); })
        .def("__neg__", [](const PyExpr& a) { return wrap(neg(a.p)); })
        .def("__pos__", [](const PyExpr& a) { return a; })
        .def("subs",
             [](const PyExpr& self, const py::dict& d) {
                 std::vector<std::pair<ExprPtr, ExprPtr>> map;
                 map.reserve(d.size());
                 for (auto item : d)
                     map.emplace_back(to_expr(item.first), to_expr(item.second));
                 return wrap(subs(self.p, map));
             },
             "子表达式替换：expr.subs({x: 1, sin(y): z})，键可为任意表达式")
        .def(py::pickle(
            [](const PyExpr& e) { return expr_getstate(e.p); },
            [](const py::list& state) { return PyExpr{expr_setstate(state)}; }))
        .def("__deepcopy__", [](const PyExpr& e, py::dict) { return e; })
        .def("__copy__", [](const PyExpr& e) { return e; })
        .def("__float__",
             [](const PyExpr& e) {
                 if (!e.p->is_constant())
                     throw py::type_error(
                         "cannot convert non-constant expression '" +
                         to_string(e.p) + "' to float; substitute its symbols "
                         "first with .subs({...})");
                 return e.p->number().to_double();
             })
        .def("diff",
             [](const PyExpr& e, const PyExpr& var) { return wrap(diff(e.p, var.p)); },
             "对 var 求偏导（与 fr.diff(expr, var) 等价的方法形式）")
        .def_property_readonly("free_symbols",
             [](const PyExpr& e) {
                 std::vector<PyExpr> out;
                 for (auto& s : free_symbols(e.p)) out.push_back(wrap(std::move(s)));
                 return out;
             },
             "表达式中的自由符号列表（按首次出现顺序）")
        .def_property_readonly("is_zero", [](const PyExpr& e) { return e.p->is_zero(); })
        .def_property_readonly("is_one", [](const PyExpr& e) { return e.p->is_one(); })
        .def_property_readonly("is_constant", [](const PyExpr& e) { return e.p->is_constant(); });

    m.def("symbol", [](const std::string& name) { return wrap(symbol(name)); });
    m.def("as_expr", [](py::handle h) { return wrap(to_expr(h)); });
    m.def("rational", [](int64_t n, int64_t d) { return wrap(rational(n, d)); });

    m.def("make_func", [](const std::string& name, const std::vector<py::object>& args) {
        std::vector<ExprPtr> a;
        a.reserve(args.size());
        for (const auto& o : args) a.push_back(to_expr(o));
        return wrap(func(name, std::move(a)));
    });

    m.def("diff", [](const PyExpr& e, const PyExpr& var) { return wrap(diff(e.p, var.p)); });

    // n 元构造：宽 Add/Mul（如 SymPy 转换）一次规范化完成，避免两两合并的 O(k²)
    m.def("add_many", [](const std::vector<py::object>& ops) {
        std::vector<ExprPtr> es;
        es.reserve(ops.size());
        for (const auto& o : ops) es.push_back(to_expr(o));
        return wrap(add(std::span<const ExprPtr>(es)));
    });
    m.def("mul_many", [](const std::vector<py::object>& ops) {
        std::vector<ExprPtr> es;
        es.reserve(ops.size());
        for (const auto& o : ops) es.push_back(to_expr(o));
        return wrap(mul(std::span<const ExprPtr>(es)));
    });

    m.def(
        "emit_llvm_ir",
        [](const std::vector<PyExpr>& exprs, const std::vector<PyExpr>& args,
           const std::string& name, bool fastmath, bool vecmath,
           const std::vector<bool>& uniform) {
            std::vector<ExprPtr> es, as;
            es.reserve(exprs.size());
            as.reserve(args.size());
            for (const auto& e : exprs) es.push_back(e.p);
            for (const auto& a : args) as.push_back(a.p);
            const Tape tape = lower(es, as);
            return LlvmTextBackend().emit(tape,
                                          KernelSpec{name, fastmath, vecmath, uniform});
        },
        py::arg("exprs"), py::arg("args"), py::arg("name") = "frontier_kernel",
        py::arg("fastmath") = false, py::arg("vecmath") = true,
        py::arg("uniform") = std::vector<bool>{});

    m.def("function_names", []() {
        const auto& reg = FuncRegistry::instance();
        std::vector<std::string> names;
        names.reserve(reg.size());
        for (size_t i = 0; i < reg.size(); ++i)
            names.push_back(reg.get(static_cast<FuncId>(i)).name);
        return names;
    });
}
