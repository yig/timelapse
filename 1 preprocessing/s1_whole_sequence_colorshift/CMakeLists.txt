cmake_minimum_required (VERSION 3.14)

project( colorshift_filter LANGUAGES CXX C)

# We want C++11
set (CMAKE_CXX_STANDARD 11)

include(FetchContent)

find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

FetchContent_Declare(
  jsoncpp
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
  GIT_TAG        1.9.4
)
# CMake 3.14+
FetchContent_MakeAvailable(jsoncpp)
# add_subdirectory(${jsoncpp_SOURCE_DIR} ${jsoncpp_BINARY_DIR})

# find_package(PkgConfig REQUIRED)
# pkg_check_modules(JSONCPP jsoncpp)
# link_libraries(${JSONCPP_LIBRARIES})
# link_directories(/usr/local/Cellar/jsoncpp/1.9.1/lib/)

message("FOUND CONFIG: ${jsoncpp_CONFIG}")
message("INCLUDE DIRS: ${jsoncpp_INCLUDE_DIRS}")
message("LIBRARIES: ${jsoncpp_LIBRARIES}")

add_executable( colorshift_filter colorshift_filter.cpp )

target_include_directories( colorshift_filter PUBLIC ${OpenCV_INCLUDE_DIRS} )
target_include_directories( colorshift_filter PUBLIC jsoncpp_static )

target_link_libraries( colorshift_filter ${OpenCV_LIBS}  Eigen3::Eigen jsoncpp_static)
