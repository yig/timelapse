cmake_minimum_required (VERSION 3.0)

project( detect_keyframe )
find_package( OpenCV REQUIRED )
find_package(jsoncpp REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${JSONCPP_INCLUDE_DIRS})
link_directories(/usr/local/Cellar/jsoncpp/1.9.1/lib/)

add_executable( detect_keyframe detect_keyframe.cpp )
target_link_libraries( detect_keyframe ${OpenCV_LIBS}  Eigen3::Eigen ${jsoncpp_LIBRARIES})
