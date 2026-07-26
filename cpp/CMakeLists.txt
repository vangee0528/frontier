set(FRONTIER_CORE_SOURCES
    src/number.cpp
    src/expr.cpp
    src/builders.cpp
    src/func_registry.cpp
    src/printer.cpp
    src/diff.cpp
    src/tape.cpp
    src/codegen_llvm_text.cpp
    src/codegen_vecmath.cpp
    src/subs.cpp
    src/visitor.cpp
)

add_library(frontier_core STATIC ${FRONTIER_CORE_SOURCES})
target_include_directories(frontier_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# /utf-8：源码为 UTF-8（含中文注释），MSVC 在非 UTF-8 代码页下必须显式声明
target_compile_options(frontier_core PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)
# VS2022 17.10+ 的 constexpr std::mutex 构造在目标机 msvcp140.dll 较旧时
# 会在首次加锁处 access violation（本机实测复现）。禁用以兼容旧 redist，
# 这是分发 Windows 二进制的标准防御。PUBLIC：传播到 tests 与 pyd。
target_compile_definitions(frontier_core PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR>
)
target_compile_options(frontier_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)
set_target_properties(frontier_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
