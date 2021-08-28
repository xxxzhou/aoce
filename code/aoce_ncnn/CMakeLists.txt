
find_package(NCNN)
find_package(Vulkan)

if(NOT NCNN_FOUND)
    message(FATAL_ERROR "NCNN not find.please set option AOCE_INSTALL_NCNN false")
endif()

include_directories(${NCNN_INCLUDE_DIRS} ${AOCE_HEADER_DIR} ${Vulkan_INCLUDE_DIR})

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(aoce_ncnn SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
target_link_libraries(aoce_ncnn aoce aoce_vulkan_extra ${NCNN_LIBRARYS})
aoce_output(aoce_ncnn)
install_aoce_module(aoce_ncnn FLASE)

if(WIN32)   
    message(STATUS "ncnn lib path: " ${NCNN_LIBRARYS})  
    message(STATUS "ncnn dll path: " ${NCNN_BINARYS})   
    file(COPY ${NCNN_BINARYS} DESTINATION "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}")    
    file(COPY ${NCNN_BINARYS} DESTINATION "${CMAKE_BINARY_DIR}/install/win/bin")
endif()
 
if(WIN32)    
    file(GLOB AOCE_NET_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../assets/net/*.*" )
    message(STATUS "aoce_net_path: " ${AOCE_NET_SOURCE})    
    file(COPY ${AOCE_NET_SOURCE} DESTINATION "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/net")   
    file(COPY ${AOCE_NET_SOURCE} DESTINATION "${CMAKE_BINARY_DIR}/install/win/bin/net")
endif()