include(AoceHelper)

add_definitions(-DAOCE_VULKAN_EXPORT_DEFINE)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # 添加用于不同平台窗口初始化的预编译指令
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR -DWIN32_LEAN_AND_MEAN)    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR -DVK_NO_PROTOTYPES)
else()
    message(FATAL_ERROR "unsupported platform.")
endif() 

if(WIN32) 
    message(STATUS "using find_package to locate vulkan") 
    find_package(Vulkan)
    message(STATUS "vulkan found = ${Vulkan_FOUND}")  
    message(STATUS "vulkan include = ${Vulkan_INCLUDE_DIR}")  
    message(STATUS "vulkan lib = ${Vulkan_LIBRARY}")    
elseif(ANDROID)   
endif() 
# 查找vulkan库引用  
include_directories(${Vulkan_INCLUDE_DIR} ${AOCE_HEADER_DIR})   

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )  
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_sub_path(vulkan AOCE_HEADER AOCE_SOURCE) 
add_sub_path(layer AOCE_HEADER AOCE_SOURCE)
if(WIN32)  
    add_sub_path(win32 AOCE_HEADER AOCE_SOURCE) 
elseif(ANDROID)  
    add_sub_path(android AOCE_HEADER AOCE_SOURCE) 
endif()

add_library(aoce_vulkan SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
if(WIN32)
    target_link_libraries(aoce_vulkan ${Vulkan_LIBRARY} aoce aoce_win)
elseif(ANDROID)
    find_library(OPENGLES3_LIBRARY GLESv2 "OpenGL ES v3.0 library")
    find_library(EGL_LIBRARY EGL "EGL 1.4 library" )
    target_link_libraries(aoce_vulkan ${Vulkan_LIBRARY} android aoce ${OPENGLES3_LIBRARY} ${EGL_LIBRARY})
endif()

aoce_output(aoce_vulkan)  
install_aoce_module(aoce_vulkan FLASE)  





