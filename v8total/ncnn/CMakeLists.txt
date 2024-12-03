cmake_minimum_required(VERSION 3.16)

project(onte)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS ${CAMKE_CXX_FLAGS})
set(OpenCV_DIR /usr/local/bin)

find_package(OpenCV REQUIRED)

# set(NCNN_INCLUDE /home/lxai/tool/ncnntool/ncnn/build/install/include/ncnn)
# set(NCNN_LIB /home/lxai/tool/ncnntool/ncnn/build/install/lib/libncnn.a)

set(ncnn_DIR /home/lxai/tool/ncnntool/ncnn/build/install/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PRIVATE})

add_executable(onte main.cpp)

# target_link_libraries(onte ${TORCH_LIBRARIES})
target_link_libraries(onte ${OpenCV_LIBS} ncnn)
