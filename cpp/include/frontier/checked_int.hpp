#pragma once

#include <cstdint>

// 可移植的 int64 checked 算术（替代 __int128，支持 MSVC x64）。
// 返回 true 表示溢出。

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace frontier::checked {

inline bool mul(int64_t a, int64_t b, int64_t* r) {
#if defined(_MSC_VER) && !defined(__clang__)
    int64_t hi;
    *r = _mul128(a, b, &hi);
    // 无溢出 ⇔ 高 64 位是低 64 位的符号扩展
    return hi != (*r >> 63);
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
