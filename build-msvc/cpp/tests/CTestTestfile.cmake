# CMake generated Testfile for 
# Source directory: H:/WorkSpace/language-Frontier/cpp/tests
# Build directory: H:/WorkSpace/language-Frontier/build-msvc/cpp/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[frontier_cpp_tests]=] "H:/WorkSpace/language-Frontier/build-msvc/cpp/tests/Debug/frontier_tests.exe")
  set_tests_properties([=[frontier_cpp_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "H:/WorkSpace/language-Frontier/cpp/tests/CMakeLists.txt;15;add_test;H:/WorkSpace/language-Frontier/cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[frontier_cpp_tests]=] "H:/WorkSpace/language-Frontier/build-msvc/cpp/tests/Release/frontier_tests.exe")
  set_tests_properties([=[frontier_cpp_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "H:/WorkSpace/language-Frontier/cpp/tests/CMakeLists.txt;15;add_test;H:/WorkSpace/language-Frontier/cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[frontier_cpp_tests]=] "H:/WorkSpace/language-Frontier/build-msvc/cpp/tests/MinSizeRel/frontier_tests.exe")
  set_tests_properties([=[frontier_cpp_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "H:/WorkSpace/language-Frontier/cpp/tests/CMakeLists.txt;15;add_test;H:/WorkSpace/language-Frontier/cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[frontier_cpp_tests]=] "H:/WorkSpace/language-Frontier/build-msvc/cpp/tests/RelWithDebInfo/frontier_tests.exe")
  set_tests_properties([=[frontier_cpp_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "H:/WorkSpace/language-Frontier/cpp/tests/CMakeLists.txt;15;add_test;H:/WorkSpace/language-Frontier/cpp/tests/CMakeLists.txt;0;")
else()
  add_test([=[frontier_cpp_tests]=] NOT_AVAILABLE)
endif()
