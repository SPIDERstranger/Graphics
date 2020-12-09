cmake_minimum_required(VERSION 3.10)
project(test)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include)

add_executable(test main.cpp )
