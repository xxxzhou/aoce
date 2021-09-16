include(AoceHelper)

# ncnntest mattingtest
set(SRC_FILE "ncnntest")

set(AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}.cpp")

if(SRC_FILE MATCHES "ncnntest")
    # 用Nsight查看vulkan计算流程必需有窗口渲染
    set(NCNN_VULKAN_WINDOW ON)    
    if(NCNN_VULKAN_WINDOW)
        message(STATUS "NCNN_VULKAN_WINDOW ON") 
        add_definitions(-DVULKAN_SHOW=1)
        add_executable(ncnntest WIN32 ${AOCE_SOURCE})
        target_link_libraries(ncnntest aoce aoce_win aoce_ncnn)        
    else()
        message(STATUS "NCNN_VULKAN_WINDOW OFF")
        add_definitions(-DVULKAN_SHOW=0)
        add_executable(ncnntest ${AOCE_SOURCE} )
        target_link_libraries(ncnntest aoce aoce_win aoce_ncnn ${OpenCV_LIBS})        
    endif()
elseif(SRC_FILE MATCHES "mattingtest")     
    add_executable(ncnntest WIN32 ${AOCE_SOURCE})
    target_link_libraries(ncnntest aoce aoce_win aoce_ncnn)
endif()

aoce_output(ncnntest) 