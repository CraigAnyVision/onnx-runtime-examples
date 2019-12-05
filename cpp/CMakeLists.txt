cmake_minimum_required(VERSION 3.10)
project(onnxruntime CXX)

find_package(CUDA REQUIRED 10.0)

set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)

set(ONNX_ROOT /opt/onnxruntime-gpu/)
set(ONNX_INC_DIR ${ONNX_ROOT}include/)
set(LIB_ONNX ${ONNX_ROOT}lib/libonnxruntime.so)

include_directories(${ONNX_INC_DIR})

set(TARGET_NAME "ort_squeezenet")
add_executable(${TARGET_NAME}  ort_squeezenet.cpp)
target_link_libraries(${TARGET_NAME} ${LIB_ONNX})
target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -pedantic)
