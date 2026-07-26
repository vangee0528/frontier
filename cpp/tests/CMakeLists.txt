add_executable(frontier_tests
    main.cpp
    test_number.cpp
    test_expr.cpp
    test_diff.cpp
    test_tape.cpp
)
target_include_directories(frontier_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
target_link_libraries(frontier_tests PRIVATE frontier_core)

if(MINGW)
    target_link_options(frontier_tests PRIVATE -static)
endif()

add_test(NAME frontier_cpp_tests COMMAND frontier_tests)
