
include(AoceHelper)
find_package(DirectX11 REQUIRED)

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
add_sub_path(dx11 AOCE_HEADER AOCE_SOURCE)
add_sub_path(window AOCE_HEADER AOCE_SOURCE)

add_definitions(-DAOCE_WIN_EXPORT_DEFINE)
include_directories(${AOCE_HEADER_DIR} ${DirectX11_INCLUDE_DIRS})

add_library(aoce_win SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
message(STATUS "dx11 libs:" ${DirectX11_LIBRARY})
target_link_libraries(aoce_win aoce ${DirectX11_LIBRARY})

aoce_output(aoce_win)
install_aoce_module(aoce_win FLASE)  