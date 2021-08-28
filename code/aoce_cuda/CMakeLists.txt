# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
# cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
# project(aoce_cuda LANGUAGES CXX CUDA)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "cuda toolkit not find.please set option AOCE_CUDA_MODULE false")
endif()
# 在使用CUDA_COMPILE cmake宏构建cuda之前使用cmake中的命令清除所有编译器标志
# set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "" )

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
add_definitions(-DAOCE_CUDA_EXPORT_DEFINE)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O3; -G; -g;-std=c++14")
# set(CUDA_SEPARABLE_COMPILATION ON)
# C++和CUDA的编译参数,可选.如果CMAKE_CXX_FLAGS和CUDA_NVCC_FLAGS有相同参数会报错
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_61,code=sm_61")
# set(CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50;-rdc=true;-use_fast_math")

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu")
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
add_sub_path(layer AOCE_HEADER AOCE_SOURCE) 
add_sub_path(cuda AOCE_HEADER AOCE_SOURCE) 

add_library(aoce_cuda SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
set_target_properties(
    aoce_cuda
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(aoce_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${AOCE_HEADER_DIR})

# https://cmake.org/cmake/help/v3.17/module/FindCUDAToolkit.html
if(WIN32)
    target_link_libraries(aoce_cuda aoce aoce_win CUDA::cudart_static CUDA::nppc)
else()
    target_link_libraries(aoce_cuda aoce CUDA::cudart_static CUDA::nppc)
endif()
#75 70 50 60
# target_compile_options(aoce_cuda PRIVATE
#   "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
#   -gencode arch=compute_50,code=sm_50
#   -gencode arch=compute_60,code=sm_60 
#   -gencode arch=compute_70,code=sm_70 
#   -gencode arch=compute_75,code=sm_75
#   >")
set_property(TARGET aoce_cuda PROPERTY CUDA_ARCHITECTURES 50 60 70 75)

aoce_output(aoce_cuda)  
install_aoce_module(aoce_cuda FLASE)   

