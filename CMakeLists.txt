set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT} -Wall -fPIC")

find_package(Eigen3 REQUIRED)
ADD_DEFINITIONS(-DEIGEN_NO_DEBUG)
include_directories(${EIGEN3_INCLUDE_DIR})

option(BUILD_PYTHON_INTERFACE OFF)
add_subdirectory("tinyfk")

include_directories("tinyfk/include")
include_directories("primitive_sdf")

option(USE_VALGRIND "Use valgrind to profile the code" OFF)
if(USE_VALGRIND)
    find_path(VALGRIND_INCLUDE_DIR
        NAMES valgrind/callgrind.h
        PATHS /usr/include /usr/local/include
    )
    if(VALGRIND_INCLUDE_DIR)
        include_directories(${VALGRIND_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "Could not find valgrind/callgrind.h. Please install the valgrind development package.")
    endif()
    add_definitions(-DUSE_VALGRIND)
endif()

file(GLOB_RECURSE SRC_FILES cpp/*.cpp)
add_subdirectory(pybind11)
pybind11_add_module(_plainmp ${SRC_FILES})
target_link_libraries(_plainmp PRIVATE tinyfk)
install(TARGETS _plainmp DESTINATION .)
