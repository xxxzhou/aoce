
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
add_sub_path(module AOCE_HEADER AOCE_SOURCE)
add_sub_path(videoDevice AOCE_HEADER AOCE_SOURCE)
add_sub_path(layer AOCE_HEADER AOCE_SOURCE)
add_sub_path(live AOCE_HEADER AOCE_SOURCE)  
add_sub_path(media AOCE_HEADER AOCE_SOURCE)   
add_sub_path(fixgraph AOCE_HEADER AOCE_SOURCE)  
add_sub_path(math AOCE_HEADER AOCE_SOURCE)  
add_sub_path(metadata AOCE_HEADER AOCE_SOURCE)  
add_sub_path(audioDevice AOCE_HEADER AOCE_SOURCE)

# message("aoce head--" ${AOCE_HEADER})
# message("aoce source--" ${AOCE_SOURCE}) 

add_definitions(-DAOCE_EXPORT_DEFINE)
add_library(aoce SHARED ${AOCE_SOURCE} ${AOCE_HEADER})
if(ANDROID)
    include_directories(${ANDROID_NDK}/sources/android/native_app_glue) 
    target_link_libraries(aoce android log)
endif()
# install的头文件
set_target_properties(aoce PROPERTIES PUBLIC_HEADER "${AOCE_HEADER}")

aoce_output(aoce) 
install_aoce_module(aoce TRUE)  
