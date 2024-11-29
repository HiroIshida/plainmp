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
option(OMPL_REGISTRATION OFF)  # avoid popup in browser
add_subdirectory(third/ompl)
include_directories("third/ompl/src")

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
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ompl/sequence_table.hpp
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ompl/generate_sequence.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ompl/generate_sequence.py
)
add_custom_target(generate_sequence_table
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ompl/sequence_table.hpp
)

file(GLOB_RECURSE SRC_FILES cpp/*.cpp)
include_directories(cpp)
add_subdirectory(pybind11)
pybind11_add_module(_plainmp ${SRC_FILES})
add_dependencies(_plainmp generate_sequence_table)
target_link_libraries(_plainmp urdfdom_model ompl ${Boost_LIBRARIES})
install(TARGETS _plainmp DESTINATION .)
