cmake_minimum_required(VERSION 3.22)
project(frontier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FRONTIER_BUILD_TESTS "Build C++ unit tests" ON)
option(FRONTIER_BUILD_PYTHON "Build the pybind11 extension module" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(cpp)

if(FRONTIER_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

if(FRONTIER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(cpp/tests)
endif()
