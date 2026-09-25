include(FetchContent)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(googletest
    # GoogleTest 1.14.0, pinned to an immutable commit.
    URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.tar.gz
    URL_HASH SHA256=7ff5db23de232a39cbb5c9f5143c355885e30ac596161a6b9fc50c4538bfbf01
)
FetchContent_MakeAvailable(googletest)

add_executable(test_motion_certificate test_motion_certificate.cpp)
target_link_libraries(test_motion_certificate PRIVATE plainmp_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_motion_certificate)
