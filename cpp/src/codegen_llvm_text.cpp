#include "frontier/codegen/llvm_text.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <set>
#include <sstream>

#include "frontier/codegen/vecmath.hpp"
#include "frontier/error.hpp"
#include "frontier/func_registry.hpp"

namespace frontier {

namespace {

// f64 常量按位十六进制打印：文本往返零精度损失
std::string f64_lit(double v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%016llX",
                  static_cast<unsigned long long>(std::bit_cast<uint64_t>(v)));
    return buf;
}

class Emitter {
public:
    Emitter(const Tape& tape, const KernelSpec& spec) : tape_(tape), spec_(spec) {}

    std::string run() {
        emit_kernel();
        std::ostringstream out;
        out << "; ModuleID = 'frontier'\n";
        out << "; 由 Frontier LlvmTextBackend 生成\n\n";
        for (const auto& d : decls_) out << d << "\n";
        if (!decls_.empty()) out << "\n";
        out << body_.str();
        for (const auto& h : used_helpers_) out << vecmath::helper_ir(h.c_str());
        return out.str();
    }

private:
    std::string fm() const { return spec_.fastmath ? "fast " : ""; }

    bool is_uniform(int32_t input_index) const {
        const auto& u = spec_.uniform_inputs;
        return input_index >= 0 && static_cast<size_t>(input_index) < u.size() &&
               u[input_index];
    }
    // Const 不物化为指令，在使用点直接内联字面量
    const std::string& v(int32_t id) const { return names_[id]; }

    void declare(const std::string& sig) { decls_.insert(sig); }

    void use_helper(const char* name) {
        if (used_helpers_.insert(name).second) {
            // 辅助函数依赖的 intrinsic
            declare("declare double @llvm.fma.f64(double, double, double)");
            declare("declare double @llvm.round.f64(double)");
            declare("declare double @llvm.minnum.f64(double, double)");
            declare("declare double @llvm.maxnum.f64(double, double)");
            declare("declare double @llvm.fabs.f64(double)");
            for (const char* dep : vecmath::helper_deps(name)) use_helper(dep);
        }
    }

    std::string call_n(const std::string& sym, const std::vector<int32_t>& args) {
        std::string sig = "declare double @" + sym + "(double";
        std::string call = "call " + fm() + "double @" + sym + "(double " + v(args[0]);
        for (size_t i = 1; i < args.size(); ++i) {
            sig += ", double";
            call += ", double " + v(args[i]);
        }
        declare(sig + ")");
        return call + ")";
    }

    // Pow 特化（见头文件说明）
    void emit_pow(int32_t id, const TapeOp& op) {
        auto& b = body_;
        const TapeOp& eop = tape_.ops[op.b];
        if (eop.kind == TapeOp::Kind::Const) {
            const double e = eop.cval;
            if (e == -1.0) {
                b << "  " << v(id) << " = fdiv " << fm() << "double " << f64_lit(1.0)
                  << ", " << v(op.a) << "\n";
                return;
            }
            if (e == 0.5) {
                declare("declare double @llvm.sqrt.f64(double)");
                b << "  " << v(id) << " = call " << fm()
                  << "double @llvm.sqrt.f64(double " << v(op.a) << ")\n";
                return;
            }
            if (e == -0.5) {
                declare("declare double @llvm.sqrt.f64(double)");
                b << "  " << v(id) << ".sq = call " << fm()
                  << "double @llvm.sqrt.f64(double " << v(op.a) << ")\n";
                b << "  " << v(id) << " = fdiv " << fm() << "double " << f64_lit(1.0)
                  << ", " << v(id) << ".sq\n";
                return;
            }
            if (e == std::nearbyint(e) && std::fabs(e) <= 8.0) {
                // 小整数指数：直接展开为乘法链（平方法），不发射 powi 调用
                // ——残留的 llvm.powi 标量调用会把循环切成向量+标量两段
                const long long n = static_cast<long long>(std::fabs(e));
                std::string acc;  // 累积乘积的名字
                std::string sq = v(op.a);
                long long bit = 0;
                for (long long k = n; k > 0; k >>= 1, ++bit) {
                    if (bit > 0) {
                        std::string nsq = v(id) + ".sq" + std::to_string(bit);
                        b << "  " << nsq << " = fmul " << fm() << "double " << sq
                          << ", " << sq << "\n";
                        sq = nsq;
                    }
                    if (k & 1) {
                        if (acc.empty()) {
                            acc = sq;
                        } else {
                            std::string nacc = v(id) + ".acc" + std::to_string(bit);
                            b << "  " << nacc << " = fmul " << fm() << "double "
                              << acc << ", " << sq << "\n";
                            acc = nacc;
                        }
                    }
                }
                if (e < 0.0) {
                    b << "  " << v(id) << " = fdiv " << fm() << "double "
                      << f64_lit(1.0) << ", " << acc << "\n";
                } else {
                    b << "  " << v(id) << " = fadd " << fm() << "double "
                      << acc << ", " << f64_lit(0.0) << "\n";
                }
                return;
            }
            if (e == std::nearbyint(e) && std::fabs(e) <= 65536.0) {
                // 大整数指数：llvm.powi
                declare("declare double @llvm.powi.f64.i32(double, i32)");
                b << "  " << v(id) << " = call " << fm()
                  << "double @llvm.powi.f64.i32(double " << v(op.a) << ", i32 "
                  << static_cast<long long>(e) << ")\n";
                return;
            }
        }
        declare("declare double @llvm.pow.f64(double, double)");
        b << "  " << v(id) << " = call " << fm() << "double @llvm.pow.f64(double "
          << v(op.a) << ", double " << v(op.b) << ")\n";
    }

    void emit_call(int32_t id, const TapeOp& op) {
        const FuncDef& def = FuncRegistry::instance().get(op.func);
        auto& b = body_;
        switch (def.codegen.lower) {
            case CodegenSpec::Lower::Intrinsic:
            case CodegenSpec::Lower::LibmCall: {
                if (op.args.empty())
                    throw CompileError("codegen: nullary call for '" + def.name + "'");
                // 向量化数学：有可内联实现的函数改走 fr_* 辅助函数
                if (spec_.vecmath) {
                    if (const char* helper = vecmath::helper_for(def.name.c_str())) {
                        use_helper(helper);
                        b << "  " << v(id) << " = call " << fm() << "double @"
                          << helper << "(double " << v(op.args[0]) << ")\n";
                        return;
                    }
                }
                b << "  " << v(id) << " = " << call_n(def.codegen.symbol, op.args)
                  << "\n";
                return;
            }
            case CodegenSpec::Lower::Custom: {
                const std::string& sym = def.codegen.symbol;
                // 比较谓词：fcmp + select 0/1（无分支，可向量化）
                if (sym == "lt" || sym == "le" || sym == "gt" || sym == "ge") {
                    const char* cc = sym == "lt"   ? "olt"
                                     : sym == "le" ? "ole"
                                     : sym == "gt" ? "ogt"
                                                   : "oge";
                    b << "  " << v(id) << ".c = fcmp " << cc << " double "
                      << v(op.args[0]) << ", " << v(op.args[1]) << "\n";
                    b << "  " << v(id) << " = select i1 " << v(id)
                      << ".c, double " << f64_lit(1.0) << ", double "
                      << f64_lit(0.0) << "\n";
                    return;
                }
                if (sym == "where") {
                    b << "  " << v(id) << ".c = fcmp one double "
                      << v(op.args[0]) << ", " << f64_lit(0.0) << "\n";
                    b << "  " << v(id) << " = select i1 " << v(id)
                      << ".c, double " << v(op.args[1]) << ", double "
                      << v(op.args[2]) << "\n";
                    return;
                }
                if (def.codegen.symbol == "sign") {
                    const std::string x = v(op.args[0]);
                    b << "  " << v(id) << ".gt = fcmp ogt double " << x << ", "
                      << f64_lit(0.0) << "\n";
                    b << "  " << v(id) << ".lt = fcmp olt double " << x << ", "
                      << f64_lit(0.0) << "\n";
                    b << "  " << v(id) << ".pos = select i1 " << v(id)
                      << ".gt, double " << f64_lit(1.0) << ", double " << f64_lit(0.0)
                      << "\n";
                    b << "  " << v(id) << " = select i1 " << v(id) << ".lt, double "
                      << f64_lit(-1.0) << ", double " << v(id) << ".pos\n";
                    return;
                }
                throw CompileError("codegen: unknown custom lowering '" +
                                   def.codegen.symbol + "'");
            }
        }
        throw CompileError("codegen: unhandled lowering kind");
    }

    void emit_kernel() {
        auto& b = body_;
        b << "define void @" << spec_.name
          << "(double** noalias nocapture readonly %ins, "
             "double** noalias nocapture readonly %outs, i64 %n) {\n";
        b << "entry:\n";

        // 预载输入/输出数组基址
        for (size_t i = 0; i < tape_.num_inputs; ++i) {
            b << "  %in" << i << ".pp = getelementptr inbounds double*, double** %ins, i64 "
              << i << "\n";
            b << "  %in" << i << " = load double*, double** %in" << i << ".pp\n";
        }
        for (size_t i = 0; i < tape_.outputs.size(); ++i) {
            b << "  %out" << i << ".pp = getelementptr inbounds double*, double** %outs, i64 "
              << i << "\n";
            b << "  %out" << i << " = load double*, double** %out" << i << ".pp\n";
        }
        // uniform 输入：批量内取同一值，entry block 载入一次
        names_.resize(tape_.ops.size());
        for (size_t id = 0; id < tape_.ops.size(); ++id) {
            const TapeOp& op = tape_.ops[id];
            if (op.kind != TapeOp::Kind::Input) continue;
            if (!is_uniform(op.input_index)) continue;
            names_[id] = "%v" + std::to_string(id);
            b << "  " << names_[id] << " = load double, double* %in"
              << op.input_index << "\n";
        }

        b << "  %empty = icmp sle i64 %n, 0\n";
        b << "  br i1 %empty, label %exit, label %loop\n\n";

        b << "loop:\n";
        b << "  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]\n";

        for (size_t id = 0; id < tape_.ops.size(); ++id) {
            const TapeOp& op = tape_.ops[id];
            const int32_t sid = static_cast<int32_t>(id);
            if (op.kind == TapeOp::Kind::Input && is_uniform(op.input_index))
                continue;  // 已在 entry 载入
            names_[id] = (op.kind == TapeOp::Kind::Const)
                             ? f64_lit(op.cval)
                             : "%v" + std::to_string(id);
            switch (op.kind) {
                case TapeOp::Kind::Input:
                    b << "  " << v(sid) << ".p = getelementptr inbounds double, double* %in"
                      << op.input_index << ", i64 %i\n";
                    b << "  " << v(sid) << " = load double, double* " << v(sid) << ".p\n";
                    break;
                case TapeOp::Kind::Const:
                    break;  // 已内联
                case TapeOp::Kind::Add:
                    b << "  " << v(sid) << " = fadd " << fm() << "double " << v(op.a)
                      << ", " << v(op.b) << "\n";
                    break;
                case TapeOp::Kind::Mul:
                    b << "  " << v(sid) << " = fmul " << fm() << "double " << v(op.a)
                      << ", " << v(op.b) << "\n";
                    break;
                case TapeOp::Kind::Pow:
                    emit_pow(sid, op);
                    break;
                case TapeOp::Kind::Call:
                    emit_call(sid, op);
                    break;
            }
        }

        for (size_t i = 0; i < tape_.outputs.size(); ++i) {
            b << "  %store" << i << ".p = getelementptr inbounds double, double* %out" << i
              << ", i64 %i\n";
            b << "  store double " << v(tape_.outputs[i]) << ", double* %store" << i
              << ".p\n";
        }

        b << "  %i.next = add nuw nsw i64 %i, 1\n";
        b << "  %done = icmp eq i64 %i.next, %n\n";
        b << "  br i1 %done, label %exit, label %loop\n\n";
        b << "exit:\n";
        b << "  ret void\n";
        b << "}\n";
    }

    const Tape& tape_;
    const KernelSpec& spec_;
    std::set<std::string> decls_;
    std::set<std::string> used_helpers_;
    std::ostringstream body_;
    std::vector<std::string> names_;
};

}  // namespace

std::string LlvmTextBackend::emit(const Tape& tape, const KernelSpec& spec) const {
    return Emitter(tape, spec).run();
}

}  // namespace frontier
