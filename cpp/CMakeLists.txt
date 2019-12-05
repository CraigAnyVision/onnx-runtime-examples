cmake_minimum_required(VERSION 3.10)
project(onnxruntime)

set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)

set(ONNX_ROOT /opt/onnxruntime/)
set(ONNX_INC_DIR ${ONNX_ROOT}include/)
set(LIB_ONNX ${ONNX_ROOT}lib/libonnxruntime.so)

include_directories(${ONNX_INC_DIR})

add_executable(ort_squeezenet ort_squeezenet.cpp)
target_link_libraries(ort_squeezenet ${LIB_ONNX})
