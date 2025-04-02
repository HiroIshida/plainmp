cmake_minimum_required(VERSION 2.6)
Project("SDFGen")

SET(CMAKE_BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
# static
add_library(sdfgen STATIC makelevelset3.cpp)
# add_executable(${PROJECT_NAME} main.cpp  makelevelset3.cpp)
