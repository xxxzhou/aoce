
include(AoceHelper)
file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_definitions(-DAOCE_WIN_EXPORT_DEFINE)
include_directories(${AOCE_HEADER_DIR})

add_library(aoce_win SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
target_link_libraries(aoce_win aoce)

aoce_output(aoce_win)
install_aoce_module(aoce_win FLASE)  