find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core module.cpp)
target_link_libraries(_core PRIVATE frontier_core)
target_compile_options(_core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

# MinGW: 静态链接运行时，避免用户环境缺 libgcc/libstdc++ DLL
if(MINGW)
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++ -static)
endif()

if(SKBUILD)
    install(TARGETS _core DESTINATION frontier)
else()
    # 开发构建：直接落到源码树的 python/frontier/，便于就地 import
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/frontier"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/frontier"
    )
endif()
