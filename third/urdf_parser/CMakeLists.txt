cmake_minimum_required( VERSION 2.8 FATAL_ERROR )
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
set (URDF_MAJOR_VERSION 1)
set (URDF_MINOR_VERSION 0)
set (URDF_PATCH_VERSION 4)

set (URDF_VERSION ${URDF_MAJOR_VERSION}.${URDF_MINOR_VERSION}.${URDF_PATCH_VERSION})
set (URDF_MAJOR_MINOR_VERSION ${URDF_MAJOR_VERSION}.${URDF_MINOR_VERSION})

add_subdirectory("tinyxml")
set(TinyXML_INCLUDE_DIRS "tinyxml")

add_library(urdfdom_model STATIC src/pose.cpp src/model.cpp src/link.cpp src/joint.cpp)
target_include_directories(urdfdom_model PUBLIC "include" ${TinyXML_INCLUDE_DIRS})
target_link_libraries(urdfdom_model tinyxml)
set_target_properties(urdfdom_model PROPERTIES SOVERSION ${URDF_MAJOR_MINOR_VERSION})
