include(AoceHelper)

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

# // 用Nsight查看vulkan计算流程必需有窗口渲染
option(NCNN_VULKAN_WINDOW "ncnntest sample show vulkan window" ON)

if(NCNN_VULKAN_WINDOW)
    message(STATUS "NCNN_VULKAN_WINDOW ON") 
    add_definitions(-DVULKAN_SHOW=1)
    add_executable(ncnntest WIN32 ${AOCE_SOURCE} ${AOCE_HEADER})
    target_link_libraries(ncnntest aoce aoce_win aoce_ncnn)
    aoce_output(ncnntest) 
else()
    message(STATUS "NCNN_VULKAN_WINDOW OFF")
    add_executable(ncnntest ${AOCE_SOURCE} ${AOCE_HEADER})
    target_link_libraries(ncnntest aoce aoce_win aoce_ncnn ${OpenCV_LIBS})
    aoce_output(ncnntest) 
endif()