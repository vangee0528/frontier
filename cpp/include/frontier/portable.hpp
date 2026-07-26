#pragma once

// 跨工具链可移植性兜底。发布矩阵的最低公分母：
//   manylinux2014 devtoolset-10 (GCC 10)  —— 无 std::bit_cast
//   macOS 部署目标 < 13.3 (AppleClang)     —— 浮点 std::to_chars 不可用
//   MSVC x64 / MinGW-w64 / GCC 11+ / Clang —— 全特性

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(__cpp_lib_bit_cast)
#include <bit>
#endif

// Apple 平台的浮点 to_chars 受 availability 限制（需要 macOS 13.3+ 部署
// 目标），为保持宽部署面直接走 snprintf 路径
#if defined(__cpp_lib_to_chars) && !defined(__APPLE__)
#define FRONTIER_HAS_FP_TO_CHARS 1
#include <charconv>
#endif

namespace frontier::portable {

inline uint64_t f64_bits(double v) {
#if defined(__cpp_lib_bit_cast)
    return std::bit_cast<uint64_t>(v);
#else
    uint64_t r;
    std::memcpy(&r, &v, sizeof r);
    return r;
#endif
}

// double 的往返安全十进制表示。to_chars 给最短形式（"0.5"）；
// 兜底 %.17g 同样保证往返，只是可能多几位数字。
inline std::string format_double(double v) {
    char buf[40];
#if FRONTIER_HAS_FP_TO_CHARS
    const auto res = std::to_chars(buf, buf + sizeof buf, v);
    return std::string(buf, res.ptr);
#else
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
#endif
}

}  // namespace frontier::portable
