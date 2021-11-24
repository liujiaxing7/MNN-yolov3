# put output to build dir
cmake_minimum_required(VERSION 3.10.2)
project(Demo)

set(CMAKE_CXX_STANDARD 14)
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)

set(MNN_DIR /home/fandong/Code/MNN)
include_directories(/home/fandong/Code/MNN/3rd_party/imageHelper)
include_directories(${MNN_DIR}/include)
include_directories(${MNN_DIR}/include/MNN)
include_directories(${MNN_DIR}/tools)
include_directories(${MNN_DIR}/tools/cpp)
include_directories(${MNN_DIR}/source)
include_directories(${MNN_DIR}/source/backend)
include_directories(${MNN_DIR}/source/core)

LINK_DIRECTORIES(${MNN_DIR}/build)

set(MNN_ROOT ${MNN_DIR}/include )
include_directories(${MNN_ROOT})
# add following lines
#aux_source_directory(./utils DIR_LIB_SRCS1)



add_executable(yolo.out yolov3.cpp)
target_link_libraries(yolo.out ${MNN_DIR})
target_link_libraries(yolo.out /home/fandong/Code/MNN/build/libMNN.so)



find_package(OpenCV REQUIRED)
target_link_libraries(yolo.out ${OpenCV_LIBS} )