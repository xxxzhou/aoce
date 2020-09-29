
# 添加子文件夹里文件并归类
function(add_sub_path relativePath HEADER_FILES SOURCE_FILELIST)
    string(REPLACE "/" "\\" filterPart ${relativePath})
    file(GLOB TEMP_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${relativePath}/*.cpp") 
    file(GLOB TEMP_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/${relativePath}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${relativePath}/*.hpp") 
    # 这是列表的操作方式    
    set(${HEADER_FILES} ${${SOURCE_FILELIST}} ${TEMP_HEADER} PARENT_SCOPE)
    set(${SOURCE_FILELIST} ${${HEADER_FILES}} ${TEMP_SOURCE} PARENT_SCOPE)
    source_group(${filterPart} FILES ${TEMP_HEADER} ${TEMP_SOURCE})   
endfunction(add_sub_path relativePath SOURCE_FILELIST HEADER_FILES)