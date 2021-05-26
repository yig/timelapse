cmake_minimum_required (VERSION 3.0)

project( colorshift_filter )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package (Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.2.0)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( colorshift_filter colorshift_filter.cpp )
target_link_libraries(colorshift_filter PUBLIC ${OpenCV_LIBS})
target_link_libraries(colorshift_filter PUBLIC Eigen3::Eigen)
target_link_libraries(colorshift_filter PRIVATE nlohmann_json::nlohmann_json)