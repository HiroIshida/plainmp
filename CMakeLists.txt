cmake_minimum_required(VERSION 3.15)
project(plainmp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT} -fPIC")

option(PLAINMP_MARCH_NATIVE "Optimize C++ code for the build machine's CPU" ON)
if(PLAINMP_MARCH_NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" PLAINMP_HAS_MARCH_NATIVE)
    if(PLAINMP_HAS_MARCH_NATIVE)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    else()
        message(WARNING "The C++ compiler does not support -march=native; building without it")
    endif()
endif()

find_package(Eigen3 REQUIRED)
ADD_DEFINITIONS(-DEIGEN_NO_DEBUG)
ADD_DEFINITIONS(-DEIGEN_DONT_VECTORIZE)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory("third/urdf_parser")
include_directories("third/urdf_parser/include")

find_package(Boost REQUIRED COMPONENTS serialization)
find_package(ompl REQUIRED)

include_directories(${OMPL_INCLUDE_DIRS})
add_subdirectory("third/ompl-ertconnect")
include_directories("third/ompl-ertconnect")

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

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cpp/plainmp/ompl/sequence_table.hpp
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/cpp/plainmp/ompl/generate_sequence.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/plainmp/ompl/generate_sequence.py
)
add_custom_target(generate_sequence_table
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/plainmp/ompl/sequence_table.hpp
)

file(GLOB_RECURSE SRC_FILES cpp/*.cpp)
include_directories(cpp)
if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()
find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE nanobind_result
)
if(NOT nanobind_result EQUAL 0)
    message(FATAL_ERROR "Could not locate nanobind for ${Python_EXECUTABLE}")
endif()
find_package(nanobind CONFIG REQUIRED)
# Match the optimization of the former pybind11 target. LTO is also needed by
# the statically linked ERTConnect planner with OMPL 1.6 and GCC 9.
nanobind_add_module(_plainmp NOMINSIZE LTO ${SRC_FILES})
add_dependencies(_plainmp generate_sequence_table)

if (TARGET ompl::ompl)
    set(_ompl_target ompl::ompl)
else()
    set(_ompl_target ompl)  # old versions of OMPL
endif()
target_link_libraries(_plainmp PRIVATE urdfdom_model ${_ompl_target} ertconnect ${Boost_LIBRARIES})
if(ompl_VERSION VERSION_LESS "1.7.0")
    target_compile_definitions(_plainmp PRIVATE OMPL_OLD_VERSION)
endif()
install(TARGETS _plainmp DESTINATION .)
