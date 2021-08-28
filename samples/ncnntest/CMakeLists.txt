include(AoceHelper)

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_executable(ncnntest ${AOCE_SOURCE} ${AOCE_HEADER})

target_link_libraries(ncnntest aoce aoce_win aoce_ncnn ${OpenCV_LIBS})
aoce_output(ncnntest) 