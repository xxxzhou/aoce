
include(AoceHelper)
file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

# #添加子文件夹里文件并归类
# function(add_sub_path relativePath)
#     string(REPLACE "/" "\\" filterPart ${relativePath})
#     file(GLOB TEMP_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${relativePath}/*.cpp") 
#     file(GLOB TEMP_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/${relativePath}/*.h"
#         "${CMAKE_CURRENT_SOURCE_DIR}/${relativePath}/*.hpp")    
#     set(AOCE_SOURCE ${AOCE_SOURCE} ${TEMP_SOURCE} PARENT_SCOPE)
#     set(AOCE_HEADER ${AOCE_HEADER} ${TEMP_HEADER} PARENT_SCOPE)
    
#     source_group(${filterPart} FILES ${TEMP_HEADER} ${TEMP_SOURCE})   
# endfunction(add_sub_path relativePath)

# 添加module
add_sub_path(Module AOCE_HEADER AOCE_SOURCE)
add_sub_path(VideoDevice AOCE_HEADER AOCE_SOURCE)
add_sub_path(Layer AOCE_HEADER AOCE_SOURCE)
add_sub_path(Live AOCE_HEADER AOCE_SOURCE)  
add_sub_path(Media AOCE_HEADER AOCE_SOURCE)  
add_sub_path(FixGraph AOCE_HEADER AOCE_SOURCE)  

message("aoce head--" ${AOCE_HEADER})
message("aoce source--" ${AOCE_SOURCE}) 

add_definitions(-DAOCE_EXPORT_DEFINE)
add_library(aoce SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
if(ANDROID)
    include_directories(${ANDROID_NDK}/sources/android/native_app_glue) 
    target_link_libraries(aoce android log unwind)
endif()
# install的头文件
set_target_properties(aoce PROPERTIES PUBLIC_HEADER "${AOCE_HEADER}")

aoce_output(aoce AOCE_HEADER) 
install_aoce_module(aoce TRUE)  
