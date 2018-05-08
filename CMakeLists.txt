cmake_minimum_required(VERSION 2.8.3)
project(detector)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

set(ARCH "sm_61")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -std=c++11 -march=native -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize -mavx2 -mfma -fomit-frame-pointer")
SET(CUDA_NVCC_FLAGS "-arch=${ARCH} --use_fast_math -O3" CACHE STRING "nvcc flags" FORCE)
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC) 
CUDA_ADD_LIBRARY(koral ${LIB_TYPE} CUDALERP.cu CLATCH.cu)

add_executable(detector_node main.cpp)
add_dependencies(detector_node koral)

target_link_libraries(detector_node
  koral
  ${OpenCV_LIBS}
 )
