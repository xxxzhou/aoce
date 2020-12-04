
add_subdirectory(aoce)
message(STATUS "system:" ${WIN32})
add_subdirectory(aoce_vulkan)
if(WIN32)
    set(AOCE_WIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/code/aoce_win")
    # dx11
    add_subdirectory(aoce_win) 
    add_subdirectory(aoce_win_mf)
    # add_subdirectory(aoce_cuda)
endif()

if(AOCE_INSTALL_AGORA)
    add_subdirectory(aoce_agora)
endif()

if(AOCE_INSTALL_FFMPEG)
    add_subdirectory(aoce_ffmpeg)
endif()

