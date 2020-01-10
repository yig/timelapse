cmake_minimum_required (VERSION 3.0)

project( layer_extraction )
find_package( OpenCV REQUIRED )
find_package(jsoncpp REQUIRED)
find_package(ZLIB REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

message("FOUND CONFIG: ${JSONCPP_CONFIG}")
message("INCLUDE DIRS: ${JSONCPP_INCLUDE_DIRS}")
message("LIBRARIES: ${JSONCPP_LIBRARIES}")

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${JSONCPP_INCLUDE_DIRS})

add_executable( layer_extraction layer_extraction.cpp )
target_link_libraries( layer_extraction ${OpenCV_LIBS}  Eigen3::Eigen ${jsoncpp_LIBRARIES} ${ZLIB_LIBRARIES})
