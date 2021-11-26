# put output to build dir
cmake_minimum_required(VERSION 3.10.2)
project(Demo)

set(CMAKE_CXX_STANDARD 14)

set(MNN_DIR /home/fandong/Downloads/MNN-1.1.2)
include_directories(${MNN_DIR}/3rd_party/imageHelper)
include_directories(${MNN_DIR}/include)
include_directories(${MNN_DIR}/include/MNN)
include_directories(${MNN_DIR}/tools)
include_directories(${MNN_DIR}/tools/cpp)
include_directories(${MNN_DIR}/source)
include_directories(${MNN_DIR}/source/backend)
include_directories(${MNN_DIR}/source/core)

LINK_DIRECTORIES(${MNN_DIR}/build-gpu)

set(MNN_ROOT ${MNN_DIR}/include )
include_directories(${MNN_ROOT})


include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(yolo.out yolov3.cpp)
target_link_libraries(yolo.out ${MNN_DIR})
target_link_libraries(yolo.out ${MNN_DIR}/build-gpu/libMNN.so)



find_package(OpenCV REQUIRED)
target_link_libraries(yolo.out ${OpenCV_LIBS} )