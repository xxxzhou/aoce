
# 所有目录的对位相对于code,导出到外部相对于统一位置
include_directories("./")

add_subdirectory(aoce) 
message(STATUS "system:" ${WIN32})
add_subdirectory(aoce_vulkan)
add_subdirectory(aoce_vulkan_extra)
if(WIN32)
    set(AOCE_WIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/code/aoce_win")
    # dx11
    add_subdirectory(aoce_win) 
    add_subdirectory(aoce_win_mf)
    if(AOCE_CUDA_MODULE)
        add_subdirectory(aoce_cuda)    
    endif()
elseif(ANDROID)
    add_subdirectory(aoce_android)
endif()

if(AOCE_INSTALL_AGORA AND AOCE_AGORA_EXIST)
    add_subdirectory(aoce_agora)
endif()

if(AOCE_INSTALL_FFMPEG)
    add_subdirectory(aoce_ffmpeg)
endif()

if(AOCE_TALKTO_EXIST)
    add_subdirectory(aoce_talkto)
endif()



