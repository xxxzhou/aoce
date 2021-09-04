include(AoceHelper)

add_definitions(-DAOCE_NCNN_EXTRA_EXPORT_DEFINE)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # 添加用于不同平台窗口初始化的预编译指令
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR -DWIN32_LEAN_AND_MEAN)    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR -DVK_NO_PROTOTYPES)
else()
    message(FATAL_ERROR "unsupported platform.")
endif()  

find_package(NCNN)
if(NOT NCNN_FOUND)
    message(FATAL_ERROR "NCNN not find.please set option AOCE_INSTALL_NCNN false")
endif()

include_directories(${NCNN_INCLUDE_DIRS} ${Vulkan_INCLUDE_DIR} ${AOCE_HEADER_DIR})

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(aoce_ncnn SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
if(WIN32)
    target_link_libraries(aoce_ncnn ${Vulkan_LIBRARY} aoce aoce_vulkan_extra ${NCNN_LIBRARYS})
elseif(ANDROID)
    target_link_libraries(aoce_ncnn ${Vulkan_LIBRARY} aoce android aoce_vulkan_extra ${NCNN_LIBRARYS})
endif()

aoce_output(aoce_ncnn)
install_aoce_module(aoce_ncnn FLASE)
  
message(STATUS "ncnn lib path: " ${NCNN_LIBRARYS})  
message(STATUS "ncnn dll path: " ${NCNN_BINARYS})  
if(WIN32)  
    file(COPY ${NCNN_BINARYS} DESTINATION "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}")    
    file(COPY ${NCNN_BINARYS} DESTINATION "${CMAKE_BINARY_DIR}/install/win/bin")
endif()
 
if(WIN32)    
    file(GLOB AOCE_NET_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../assets/net/*.*" )
    message(STATUS "aoce_net_path: " ${AOCE_NET_SOURCE})    
    file(COPY ${AOCE_NET_SOURCE} DESTINATION "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/net")   
    file(COPY ${AOCE_NET_SOURCE} DESTINATION "${CMAKE_BINARY_DIR}/install/win/bin/net")
endif()