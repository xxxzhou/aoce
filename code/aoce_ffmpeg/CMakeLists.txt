
find_package(FFmpeg)

if(NOT FFmpeg_FOUND)
    message(FATAL_ERROR "FFmpeg not find.please set option AOCE_INSTALL_FFMPEG false")
endif()

include_directories(${FFMPEG_INCLUDE_DIRS} ${AOCE_HEADER_DIR})

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_sub_path(media AOCE_HEADER AOCE_SOURCE)

add_library(aoce_ffmpeg SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
target_link_libraries(aoce_ffmpeg aoce ${FFMPEG_LIBRARIES})
aoce_output(aoce_ffmpeg)
install_aoce_module(aoce_ffmpeg FLASE)

if(WIN32)  
    message(STATUS "ffmpeg dll path: " ${FFMPEG_BINARYS})  
    file(COPY ${FFMPEG_BINARYS} DESTINATION "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}")    
endif()