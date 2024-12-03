cmake_minimum_required(VERSION 3.16)
project(onte)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR /usr/local/lib)

find_package(OpenCV REQUIRED)

set(OPENVINO_INCLUDE_DIR /home/lxai/tool/openvinotool/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64/runtime/include)
set(OPENVINO_LIB /home/lxai/tool/openvinotool/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64/runtime/lib/intel64/libopenvino.so)

include_directories(${OpenCV_INCLUDE_DIRS} ${OPENVINO_INCLUDE_DIR})
add_executable(onte main.cpp)

target_link_libraries(onte ${OpenCV_LIBS} ${OPENVINO_LIB})
