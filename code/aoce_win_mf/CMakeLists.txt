

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_sub_path(MFVideoDevice AOCE_HEADER AOCE_SOURCE)

add_library(aoce_win_mf SHARED ${AOCE_SOURCE} ${AOCE_HEADER})

message(STATUS "mf- " ${AOCE_HEADER_DIR})

include_directories(${AOCE_HEADER_DIR} ${AOCE_WIN_HEADER_DIR})
# link_libraries

target_link_libraries(aoce_win_mf aoce aoce_win mf mfplat mfreadwrite mfuuid shlwapi)

aoce_output(aoce_win_mf)
install_aoce_module(aoce_win_mf FLASE)  