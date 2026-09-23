cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT} -fPIC")

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
option(PLAINMP_ENABLE_AVX2_COLLISION "Batch collision checks across four states on AVX2 CPUs" ON)
set(_batch_avx2_source ${CMAKE_CURRENT_SOURCE_DIR}/cpp/plainmp/constraints/sphere_collision_batch_avx2.cpp)
if(PLAINMP_ENABLE_AVX2_COLLISION AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$"
   AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    # Keep the ISA-specific kernel out of LTO, so portable callers never acquire
    # AVX instructions before the runtime CPU feature check.
    set_source_files_properties(${_batch_avx2_source} PROPERTIES
        COMPILE_OPTIONS "-mavx2;-ffp-contract=off;-fno-lto")
    set(_batch_avx2_enabled TRUE)
else()
    list(REMOVE_ITEM SRC_FILES ${_batch_avx2_source})
endif()
include_directories(cpp)
add_subdirectory(pybind11)
pybind11_add_module(_plainmp ${SRC_FILES})
add_dependencies(_plainmp generate_sequence_table)
if(_batch_avx2_enabled)
    target_compile_definitions(_plainmp PRIVATE PLAINMP_HAS_AVX2_COLLISION PLAINMP_BATCH_STATES)
endif()

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

option(PLAINMP_BUILD_BATCH_QUERY_CHECK "Build diagnostic bindings for batch query checks" OFF)
if(PLAINMP_BUILD_BATCH_QUERY_CHECK)
    target_sources(_plainmp PRIVATE example/bench/check_batch_queries.cpp)
    target_compile_definitions(_plainmp PRIVATE PLAINMP_BATCH_QUERY_CHECK)
endif()

option(PLAINMP_BUILD_KDTREE_CHECK "Build the standalone KD-tree regression check" OFF)
if(PLAINMP_BUILD_KDTREE_CHECK)
    add_executable(check_kdtree example/bench/check_kdtree.cpp cpp/plainmp/collision/kdtree.cpp)
endif()

option(PLAINMP_BUILD_BATCH_COLLISION_CHECK "Build the standalone batch collision regression check" OFF)
if(PLAINMP_BUILD_BATCH_COLLISION_CHECK)
    set(_batch_check_sources
        example/bench/check_batched_collision.cpp
        cpp/plainmp/kinematics/algorithm.cpp
        cpp/plainmp/kinematics/kinematics.cpp
        cpp/plainmp/kinematics/kinematic_model_wrapper.cpp
        cpp/plainmp/collision/kdtree.cpp
        cpp/plainmp/constraints/primitive_sphere_collision.cpp
        cpp/plainmp/constraints/sphere_collision_batch.cpp)
    if(_batch_avx2_enabled)
        list(APPEND _batch_check_sources ${_batch_avx2_source})
    endif()
    add_executable(check_batched_collision ${_batch_check_sources})
    target_link_libraries(check_batched_collision PRIVATE urdfdom_model)
    if(_batch_avx2_enabled)
        target_compile_definitions(check_batched_collision PRIVATE PLAINMP_HAS_AVX2_COLLISION)
    endif()
endif()
