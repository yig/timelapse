cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


project( detect_keyframe )
find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.2.0)


add_executable( detect_keyframe detect_keyframe.cpp )
target_link_libraries(detect_keyframe PUBLIC ${OpenCV_LIBS})
target_link_libraries(detect_keyframe PUBLIC Eigen3::Eigen)
target_link_libraries(detect_keyframe PRIVATE nlohmann_json::nlohmann_json)
