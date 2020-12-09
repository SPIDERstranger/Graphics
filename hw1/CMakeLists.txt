cmake_minimum_required(VERSION 3.10)
project(build)

find_package(OpenCV REQUIRED)


message(STATUS "OpenCV library status:")
message(STATUS " config: ${OPENCV_DIR}")
message(STATUS " version: ${OpenV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include_path: ${OpenCV_INCLUDE_DIRS}")


set(CMAKE_CXX_STANDARD 17)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(build main.cpp rasterizer.hpp rasterizer.cpp Triangle.hpp Triangle.cpp)

link_directories(${OpenCV_LIBRARY_DIRS})
target_link_libraries(build ${OpenCV_LIBRARIES})