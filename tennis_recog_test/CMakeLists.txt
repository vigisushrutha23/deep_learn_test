project(tennis_racket_recog)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
add_subdirectory(tiny_dnn)
find_package(
  OpenCV 
 )
add_compile_options(-std=c++11)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
include_directories(${CMAKE_SOURCE_DIR}/tiny_dnn)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories (${OpenCV_INCLUDE_DIRS})
add_executable(tennis_trainer tennis_train.cpp)
target_link_libraries(tennis_trainer tiny_dnn ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(tennis_recog tennis_recog.cpp)
target_link_libraries(tennis_recog tiny_dnn ${OpenCV_LIBS} ${Boost_LIBRARIES})
