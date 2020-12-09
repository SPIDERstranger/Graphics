cmake_minimum_required(VERSION 3.10)
project(Program)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(/usr/local/include)

add_executable(Program main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp)
target_link_libraries(Program ${OpenCV_LIBRARIES})
