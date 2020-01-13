cmake_minimum_required (VERSION 3.0)

project( colorshift_filter )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package( OpenCV REQUIRED )
find_package(jsoncpp REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})
link_directories(/usr/local/Cellar/jsoncpp/1.9.1/lib/)

message("FOUND CONFIG: ${JSONCPP_CONFIG}")
message("INCLUDE DIRS: ${JSONCPP_INCLUDE_DIRS}")
message("LIBRARIES: ${JSONCPP_LIBRARIES}")

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${JSONCPP_INCLUDE_DIRS})

add_executable( colorshift_filter colorshift_filter.cpp )
target_link_libraries( colorshift_filter ${OpenCV_LIBS}  Eigen3::Eigen ${jsoncpp_LIBRARIES})
