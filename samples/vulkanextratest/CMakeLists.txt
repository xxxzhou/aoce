include(AoceHelper)
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