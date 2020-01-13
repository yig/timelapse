cmake_minimum_required (VERSION 3.0)
project( moving_median_with_mask_function )
find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")


find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( moving_median_with_mask_function moving_median_with_mask_function.cpp )
target_link_libraries( moving_median_with_mask_function ${OpenCV_LIBS}  Eigen3::Eigen )
