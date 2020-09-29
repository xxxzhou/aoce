
add_subdirectory(aoce)

message(STATUS "system:" ${WIN32})

if(WIN32)
set(AOCE_WIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/code/aoce_win")
add_subdirectory(aoce_win)
add_subdirectory(aoce_win_mf)
endif()
