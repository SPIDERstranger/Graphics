cmake_minimum_required(VERSION 3.10)
project(Program)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_executable(Program main.cpp)

target_link_libraries(Program ${OpenCV_LIBRARIES})
