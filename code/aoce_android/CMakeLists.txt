# https://www.sisik.eu/blog/android/ndk/camera
# camera2ndk 
# mediandk
include(AoceHelper)

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )  
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

include_directories(${AOCE_HEADER_DIR})

find_library(camera2ndk-lib camera2ndk) 
find_library(mediandk-lib mediandk)
add_library(aoce_android SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
target_link_libraries(aoce_android ${camera2ndk-lib} ${mediandk-lib} jnigraphics android aoce)