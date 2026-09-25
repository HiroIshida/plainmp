cmake_minimum_required(VERSION 3.15)
project(plainmp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
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

file(GLOB_RECURSE CORE_FILES CONFIGURE_DEPENDS cpp/plainmp/*.cpp)
list(FILTER CORE_FILES EXCLUDE REGEX "/bindings/")
file(GLOB BINDING_FILES CONFIGURE_DEPENDS cpp/plainmp/bindings/*.cpp)
include_directories(cpp)
add_library(plainmp_core STATIC ${CORE_FILES})
set_target_properties(plainmp_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(plainmp_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
add_dependencies(plainmp_core generate_sequence_table)

# Preserve Release LTO when moving the implementation out of the Python module.
include(CheckIPOSupported)
check_ipo_supported(RESULT plainmp_ipo_supported)
if(plainmp_ipo_supported)
    set_property(TARGET plainmp_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

option(PLAINMP_ENABLE_INTERVAL_PRUNING "Experimental scalar motion-interval pruning" OFF)
if(PLAINMP_ENABLE_INTERVAL_PRUNING)
    target_compile_definitions(plainmp_core PUBLIC PLAINMP_ENABLE_INTERVAL_PRUNING)
endif()

if (TARGET ompl::ompl)
    set(_ompl_target ompl::ompl)
else()
    set(_ompl_target ompl)  # old versions of OMPL
endif()
target_link_libraries(plainmp_core PUBLIC urdfdom_model ${_ompl_target} ertconnect ${Boost_LIBRARIES})
if(ompl_VERSION VERSION_LESS "1.7.0")
    target_compile_definitions(plainmp_core PUBLIC OMPL_OLD_VERSION)
endif()

option(PLAINMP_BUILD_PYTHON "Build the Python extension" ON)
if(PLAINMP_BUILD_PYTHON)
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
    nanobind_add_module(_plainmp NOMINSIZE LTO ${BINDING_FILES})
    target_link_libraries(_plainmp PRIVATE plainmp_core)
    install(TARGETS _plainmp DESTINATION .)
endif()

# Packaging does not need GoogleTest or a network download.
option(BUILD_TESTING "Build C++ tests" OFF)
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()
