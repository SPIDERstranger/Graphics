cmake_minimum_required (VERSION 2.8.11)
project (Program)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# find_package(Eigen3 REQUIRED)
# include_directories(EIGEN3_INCLUDE_DIR)

add_executable (Program main.cpp)
target_link_libraries(Program ${OpenCV_LIBS})

