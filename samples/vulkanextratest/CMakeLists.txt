include(AoceHelper)

add_definitions(-DAOCE_VULKAN_EXTRA_EXPORT_DEFINE)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # 添加用于不同平台窗口初始化的预编译指令
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR -DWIN32_LEAN_AND_MEAN)    
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR -DVK_NO_PROTOTYPES)
else()
    message(FATAL_ERROR "unsupported platform.")
endif()  

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

if(WIN32)
    add_executable(vulkanextratest WIN32 ${AOCE_SOURCE} ${AOCE_HEADER})
else()
    add_library(vulkanextratest SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
endif() 

target_link_libraries(vulkanextratest aoce aoce_vulkan aoce_vulkan_extra)

aoce_output(vulkanextratest) 
install_aoce_module(vulkanextratest FALSE) 