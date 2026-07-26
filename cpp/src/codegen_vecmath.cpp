// sin/cos 的多项式系数与 Cody-Waite 规约常数、log 的 atanh 级数系数
// 派生自 fdlibm（Freely Distributable LIBM）：
//   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//   Developed at SunSoft, a Sun Microsystems, Inc. business.
//   Permission to use, copy, modify, and distribute this software is
//   freely granted, provided that this notice is preserved.
// 完整声明见仓库根目录 THIRD_PARTY_NOTICES.md。

#include "frontier/codegen/vecmath.hpp"

#include <cstring>
#include <string>

namespace frontier::vecmath {

namespace {

// ---------------------------------------------------------------------------
// exp：x = n·ln2 + r（Cody-Waite 两段），exp(r) 泰勒至 r^13，2^n 位构造。
// 溢出/下溢/NaN 由末尾 select 钳制。
// ---------------------------------------------------------------------------
const char* kExpIR = R"IR(
define internal double @fr_exp(double %x) alwaysinline {
  %t0 = fmul double %x, 0x3FF71547652B82FE
  %t1 = call double @llvm.minnum.f64(double %t0, double 0x4090000000000000)
  %t  = call double @llvm.maxnum.f64(double %t1, double 0xC090CC0000000000)
  %n  = call double @llvm.round.f64(double %t)
  %r1 = call double @llvm.fma.f64(double %n, double 0xBFE62E42FEE00000, double %x)
  %r  = call double @llvm.fma.f64(double %n, double 0xBDEA39EF35793C76, double %r1)
  %p0 = call double @llvm.fma.f64(double 0x3DE6124613A86D09, double %r, double 0x3E21EED8EFF8D898)
  %p1 = call double @llvm.fma.f64(double %p0, double %r, double 0x3E5AE64567F544E4)
  %p2 = call double @llvm.fma.f64(double %p1, double %r, double 0x3E927E4FB7789F5C)
  %p3 = call double @llvm.fma.f64(double %p2, double %r, double 0x3EC71DE3A556C734)
  %p4 = call double @llvm.fma.f64(double %p3, double %r, double 0x3EFA01A01A01A01A)
  %p5 = call double @llvm.fma.f64(double %p4, double %r, double 0x3F2A01A01A01A01A)
  %p6 = call double @llvm.fma.f64(double %p5, double %r, double 0x3F56C16C16C16C17)
  %p7 = call double @llvm.fma.f64(double %p6, double %r, double 0x3F81111111111111)
  %p8 = call double @llvm.fma.f64(double %p7, double %r, double 0x3FA5555555555555)
  %p9 = call double @llvm.fma.f64(double %p8, double %r, double 0x3FC5555555555555)
  %pa = call double @llvm.fma.f64(double %p9, double %r, double 0x3FE0000000000000)
  %pb = call double @llvm.fma.f64(double %pa, double %r, double 0x3FF0000000000000)
  %p  = call double @llvm.fma.f64(double %pb, double %r, double 0x3FF0000000000000)
  ; 2^n 拆成 2^(n/2)·2^(n-n/2)：单次位构造在 |n|>1022 时 biased 指数
  ; 越界产生垃圾；两半各自在正常范围内，且经硬件舍入获得正确的渐进下溢
  ; （exp(-714) 等次正规结果）
  %ni = fptosi double %n to i64
  %n1 = ashr i64 %ni, 1
  %n2 = sub nsw i64 %ni, %n1
  %nb1 = add nsw i64 %n1, 1023
  %sb1 = shl i64 %nb1, 52
  %sc1 = bitcast i64 %sb1 to double
  %nb2 = add nsw i64 %n2, 1023
  %sb2 = shl i64 %nb2, 52
  %sc2 = bitcast i64 %sb2 to double
  %r00 = fmul double %p, %sc1
  %r0 = fmul double %r00, %sc2
  %isbig = fcmp ogt double %x, 0x40862E42FEFA39EF
  %ra = select i1 %isbig, double 0x7FF0000000000000, double %r0
  %issml = fcmp olt double %x, 0xC0874910D52D3052
  %rb = select i1 %issml, double 0x0000000000000000, double %ra
  %isnan = fcmp uno double %x, %x
  %res = select i1 %isnan, double %x, double %rb
  ret double %res
}
)IR";

// ---------------------------------------------------------------------------
// sin/cos：n = round(x·2/π)，r 两段 Cody-Waite 规约至 [-π/4, π/4]，
// 象限 k 用 select 无分支选择 ±sin_poly/±cos_poly（fdlibm 系数）。
// cos 与 sin 共享结构，仅象限号 +1。
// ---------------------------------------------------------------------------
const char* kSinCosCommon = R"IR(
  %t0 = fmul double %x, 0x3FE45F306DC9C883
  %t1 = call double @llvm.minnum.f64(double %t0, double 0x432FF973CAFA8000)
  %t  = call double @llvm.maxnum.f64(double %t1, double 0xC32FF973CAFA8000)
  %n  = call double @llvm.round.f64(double %t)
  %r1 = call double @llvm.fma.f64(double %n, double 0xBFF921FB54400000, double %x)
  %r  = call double @llvm.fma.f64(double %n, double 0xBDD0B4611A626331, double %r1)
  %k  = fptosi double %n to i64
  %z  = fmul double %r, %r
  %s0 = call double @llvm.fma.f64(double 0x3DE5D93A5ACFD57C, double %z, double 0xBE5AE5E68A2B9CEB)
  %s1 = call double @llvm.fma.f64(double %s0, double %z, double 0x3EC71DE357B1FE7D)
  %s2 = call double @llvm.fma.f64(double %s1, double %z, double 0xBF2A01A019C161D5)
  %s3 = call double @llvm.fma.f64(double %s2, double %z, double 0x3F8111111110F8A6)
  %s4 = call double @llvm.fma.f64(double %s3, double %z, double 0xBFC5555555555549)
  %rz = fmul double %r, %z
  %sinp = call double @llvm.fma.f64(double %rz, double %s4, double %r)
  %c0 = call double @llvm.fma.f64(double 0xBDA8FAE9BE8838D4, double %z, double 0x3E21EE9EBDB4B1C4)
  %c1 = call double @llvm.fma.f64(double %c0, double %z, double 0xBE927E4F809C52AD)
  %c2 = call double @llvm.fma.f64(double %c1, double %z, double 0x3EFA01A019CB1590)
  %c3 = call double @llvm.fma.f64(double %c2, double %z, double 0xBF56C16C16C15177)
  %c4 = call double @llvm.fma.f64(double %c3, double %z, double 0x3FA555555555554C)
  %zz = fmul double %z, %z
  %hz = call double @llvm.fma.f64(double %z, double 0xBFE0000000000000, double 0x3FF0000000000000)
  %cosp = call double @llvm.fma.f64(double %zz, double %c4, double %hz)
)IR";

const char* kSinHead = R"IR(
define internal double @fr_sin(double %x) alwaysinline {
)IR";
const char* kSinTail = R"IR(
  %kb0 = and i64 %k, 1
  %e0 = icmp eq i64 %kb0, 0
  %base = select i1 %e0, double %sinp, double %cosp
  %kb1 = and i64 %k, 2
  %ne1 = icmp ne i64 %kb1, 0
  %nbase = fneg double %base
  %sres = select i1 %ne1, double %nbase, double %base
  %isnan = fcmp uno double %x, %x
  %res = select i1 %isnan, double %x, double %sres
  ret double %res
}
)IR";

const char* kCosHead = R"IR(
define internal double @fr_cos(double %x) alwaysinline {
)IR";
const char* kCosTail = R"IR(
  %k2 = add i64 %k, 1
  %kb0 = and i64 %k2, 1
  %e0 = icmp eq i64 %kb0, 0
  %base = select i1 %e0, double %sinp, double %cosp
  %kb1 = and i64 %k2, 2
  %ne1 = icmp ne i64 %kb1, 0
  %nbase = fneg double %base
  %sres = select i1 %ne1, double %nbase, double %base
  %isnan = fcmp uno double %x, %x
  %res = select i1 %isnan, double %x, double %sres
  ret double %res
}
)IR";

// ---------------------------------------------------------------------------
// log：位操作提取指数、尾数归一到 [√2/2, √2)，fdlibm atanh 级数。
// log(0)=-inf，log(<0)=NaN；次正规数输入不支持（文档承诺）。
// ---------------------------------------------------------------------------
const char* kLogIR = R"IR(
define internal double @fr_log(double %x) alwaysinline {
  %bits = bitcast double %x to i64
  %eraw = lshr i64 %bits, 52
  %ebias = sub nsw i64 %eraw, 1023
  %mant = and i64 %bits, 4503599627370495
  %mb = or i64 %mant, 4607182418800017408
  %m0 = bitcast i64 %mb to double
  %big = fcmp ogt double %m0, 0x3FF6A09E667F3BCD
  %m1 = fmul double %m0, 0x3FE0000000000000
  %m = select i1 %big, double %m1, double %m0
  %eadj = zext i1 %big to i64
  %e0 = add nsw i64 %ebias, %eadj
  %e = sitofp i64 %e0 to double
  %f = fsub double %m, 0x3FF0000000000000
  %fp2 = fadd double %f, 0x4000000000000000
  %s = fdiv double %f, %fp2
  %z = fmul double %s, %s
  %w = fmul double %z, %z
  %ta = call double @llvm.fma.f64(double 0x3FC39A09D078C69F, double %w, double 0x3FCC71C51D8E78AF)
  %t1 = call double @llvm.fma.f64(double %ta, double %w, double 0x3FD999999997FA04)
  %t1w = fmul double %t1, %w
  %tb = call double @llvm.fma.f64(double 0x3FC2F112DF3E5244, double %w, double 0x3FC7466496CB03DE)
  %tc = call double @llvm.fma.f64(double %tb, double %w, double 0x3FD2492494229359)
  %t2 = call double @llvm.fma.f64(double %tc, double %w, double 0x3FE5555555555593)
  %t2z = fmul double %t2, %z
  %R = fadd double %t1w, %t2z
  %ff = fmul double %f, %f
  %hfsq = fmul double %ff, 0x3FE0000000000000
  %hr = fadd double %hfsq, %R
  %shr = fmul double %s, %hr
  %el = fmul double %e, 0x3DEA39EF35793C76
  %a = fadd double %shr, %el
  %b = fsub double %hfsq, %a
  %c = fsub double %b, %f
  %eh = fmul double %e, 0x3FE62E42FEE00000
  %r0 = fsub double %eh, %c
  %eq0 = fcmp oeq double %x, 0x0000000000000000
  %ra = select i1 %eq0, double 0xFFF0000000000000, double %r0
  %lt0 = fcmp olt double %x, 0x0000000000000000
  %rb = select i1 %lt0, double 0x7FF8000000000000, double %ra
  %isinf = fcmp oeq double %x, 0x7FF0000000000000
  %rc = select i1 %isinf, double 0x7FF0000000000000, double %rb
  %isnan = fcmp uno double %x, %x
  %res = select i1 %isnan, double %x, double %rc
  ret double %res
}
)IR";

// ---------------------------------------------------------------------------
// tan = fr_sin / fr_cos（~2-3 ulp；极点附近与 sin/cos 舍入一致）
// ---------------------------------------------------------------------------
const char* kTanIR = R"IR(
define internal double @fr_tan(double %x) alwaysinline {
  %s = call double @fr_sin(double %x)
  %c = call double @fr_cos(double %x)
  %r = fdiv double %s, %c
  ret double %r
}
)IR";

// ---------------------------------------------------------------------------
// tanh：|x|<0.05 用奇次泰勒（截断误差 <1e-16），否则 (1-t)/(1+t)，
// t=exp(-2|x|)；符号用位操作回贴。NaN 走公式路径自然传播。
// ---------------------------------------------------------------------------
const char* kTanhIR = R"IR(
define internal double @fr_tanh(double %x) alwaysinline {
  %ax = call double @llvm.fabs.f64(double %x)
  %m2x = fmul double %ax, 0xC000000000000000
  %t = call double @fr_exp(double %m2x)
  %num = fsub double 0x3FF0000000000000, %t
  %den = fadd double 0x3FF0000000000000, %t
  %mag = fdiv double %num, %den
  %mbits = bitcast double %mag to i64
  %xbits = bitcast double %x to i64
  %sbits = and i64 %xbits, -9223372036854775808
  %rbits = or i64 %mbits, %sbits
  %big = bitcast i64 %rbits to double
  %u = fmul double %x, %x
  %p0 = call double @llvm.fma.f64(double 0x3F9664F4882C10FA, double %u, double 0xBFABA1BA1BA1BA1C)
  %p1 = call double @llvm.fma.f64(double %p0, double %u, double 0x3FC1111111111111)
  %p2 = call double @llvm.fma.f64(double %p1, double %u, double 0xBFD5555555555555)
  %p3 = call double @llvm.fma.f64(double %p2, double %u, double 0x3FF0000000000000)
  %small = fmul double %x, %p3
  %issmall = fcmp olt double %ax, 0x3FA999999999999A
  %res = select i1 %issmall, double %small, double %big
  ret double %res
}
)IR";

std::string build(const char* head, const char* common, const char* tail) {
    return std::string(head) + common + tail;
}

// 静态存储拼好的 sin/cos 文本
const std::string kSinIR = build(kSinHead, kSinCosCommon, kSinTail);
const std::string kCosIR = build(kCosHead, kSinCosCommon, kCosTail);

}  // namespace

const char* helper_for(const char* name) {
    if (std::strcmp(name, "sin") == 0) return "fr_sin";
    if (std::strcmp(name, "cos") == 0) return "fr_cos";
    if (std::strcmp(name, "exp") == 0) return "fr_exp";
    if (std::strcmp(name, "log") == 0) return "fr_log";
    if (std::strcmp(name, "tan") == 0) return "fr_tan";
    if (std::strcmp(name, "tanh") == 0) return "fr_tanh";
    return nullptr;
}

const char* helper_ir(const char* helper) {
    if (std::strcmp(helper, "fr_sin") == 0) return kSinIR.c_str();
    if (std::strcmp(helper, "fr_cos") == 0) return kCosIR.c_str();
    if (std::strcmp(helper, "fr_exp") == 0) return kExpIR;
    if (std::strcmp(helper, "fr_log") == 0) return kLogIR;
    if (std::strcmp(helper, "fr_tan") == 0) return kTanIR;
    if (std::strcmp(helper, "fr_tanh") == 0) return kTanhIR;
    return nullptr;
}

std::vector<const char*> helper_deps(const char* helper) {
    if (std::strcmp(helper, "fr_tan") == 0) return {"fr_sin", "fr_cos"};
    if (std::strcmp(helper, "fr_tanh") == 0) return {"fr_exp"};
    return {};
}

}  // namespace frontier::vecmath
