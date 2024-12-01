cmake_minimum_required(VERSION 3.0.2)
add_compile_options(-std=c++14)

find_package(Eigen3 REQUIRED)
ADD_DEFINITIONS(-DEIGEN_NO_DEBUG)
include_directories(${EIGEN3_INCLUDE_DIR})
 
find_package(ompl REQUIRED)
include_directories(${OMPL_INCLUDE_DIRS})

file(GLOB_RECURSE ERTTREE_SRC ertconnect/*.cpp)
include_directories(ertconnect)

add_library(ertconnect STATIC ${ERTTREE_SRC})
target_link_libraries(ertconnect ${OMPL_LIBRARIES})
