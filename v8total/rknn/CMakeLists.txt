cmake_minimum_required(VERSION 3.16)

project(onte)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR /home/ubuntu/wrwork/rknn-toolkit2/rknpu2/examples/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)

set(RKNN_DIRS /home/ubuntu/wrwork/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include)
set(RKNN_LIBS /home/ubuntu/wrwork/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${RKNN_DIRS})

add_executable(onte main.cpp)

target_link_libraries(onte ${OpenCV_LIBS} ${RKNN_LIBS})
