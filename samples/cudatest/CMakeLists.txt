
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-O3; -G; -g;-std=c++14")

include_directories("../../code")
include_directories("../../code/aoce_cuda")

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_executable(cudatest ${AOCE_SOURCE} )
set_target_properties(
    cudatest
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cudatest aoce aoce_cuda aoce_talkto_cuda ${OpenCV_LIBS} CUDA::cudart_static CUDA::nppc CUDA::nppif CUDA::npps)

target_compile_options(cudatest PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_60,code=sm_60 
  -gencode arch=compute_70,code=sm_70 
  -gencode arch=compute_75,code=sm_75
  >")

aoce_output(cudatest)

# 复制图片到运行目录下
file(GLOB AOCE_IMAGES_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../assets/images/*.bmp" )
message(STATUS "aoce_image_path: " ${AOCE_IMAGES_SOURCE})    
file(COPY ${AOCE_IMAGES_SOURCE} DESTINATION "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/images")    