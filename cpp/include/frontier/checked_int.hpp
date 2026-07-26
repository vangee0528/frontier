#pragma once

#include <cstdint>

// 可移植的 int64 checked 算术（替代 __int128，支持 MSVC x64）。
// 返回 true 表示溢出。

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace frontier::checked {

inline bool mul(int64_t a, int64_t b, int64_t* r) {
#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
    int64_t hi;
    *r = _mul128(a, b, &hi);
    // 无溢出 ⇔ 高 64 位是低 64 位的符号扩展
    return hi != (*r >> 63);
#elif defined(_MSC_VER) && !defined(__clang__)
    // 无 128 位乘法内建的 MSVC 架构（x86/ARM64 等）：先界检后乘。
    // INT64_MIN 参与时直接按溢出处理（调用方降级 Real，正确性不受影响）
    if (a == 0 || b == 0) {
        *r = 0;
        return false;
    }
    if (a == INT64_MIN || b == INT64_MIN) return true;
    const int64_t aa = a < 0 ? -a : a;
    const int64_t ab = b < 0 ? -b : b;
    if (aa > INT64_MAX / ab) return true;
    *r = a * b;
    return false;
#else
    return __builtin_mul_overflow(a, b, r);
#endif
}

inline bool add(int64_t a, int64_t b, int64_t* r) {
#if defined(_MSC_VER) && !defined(__clang__)
    const uint64_t ur = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
    *r = static_cast<int64_t>(ur);
    // 同号相加变号 ⇔ 溢出
    return ((a ^ *r) & (b ^ *r)) < 0;
#else
    return __builtin_add_overflow(a, b, r);
#endif
}

inline bool sub(int64_t a, int64_t b, int64_t* r) {
#if defined(_MSC_VER) && !defined(__clang__)
    const uint64_t ur = static_cast<uint64_t>(a) - static_cast<uint64_t>(b);
    *r = static_cast<int64_t>(ur);
    return ((a ^ b) & (a ^ *r)) < 0;
#else
    return __builtin_sub_overflow(a, b, r);
#endif
}

inline bool neg(int64_t a, int64_t* r) { return sub(0, a, r); }

}  // namespace frontier::checked
