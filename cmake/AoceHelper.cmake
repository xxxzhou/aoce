
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

# Construct search paths for includes and libraries from a PREFIX_PATH
macro(create_search_paths PREFIX)
  foreach(dir ${${PREFIX}_PREFIX_PATH})
    if(WIN32)
      set(platform_dir ${dir}/x64)
    elseif(ANDROID)
      set(platform_dir ${dir}/armeabi-v7a)
    endif()    
    set(${PREFIX}_INC_SEARCH_PATH ${${PREFIX}_INC_SEARCH_PATH}
      ${platform_dir}/include ${platform_dir}/Include ${platform_dir}/include/${PREFIX} ${platform_dir}/Headers)
    set(${PREFIX}_INC_SEARCH_PATH ${${PREFIX}_INC_SEARCH_PATH}
      ${dir}/include ${dir}/Include ${dir}/include/${PREFIX} ${dir}/Headers)
    set(${PREFIX}_LIB_SEARCH_PATH ${${PREFIX}_LIB_SEARCH_PATH}
      ${platform_dir}/lib ${platform_dir}/Lib ${platform_dir}/lib/${PREFIX} ${platform_dir}/Libs)
    set(${PREFIX}_BIN_SEARCH_PATH ${${PREFIX}_BIN_SEARCH_PATH} ${platform_dir}/dll ${platform_dir}/bin )
  endforeach(dir)
endmacro(create_search_paths)

macro(start_android_find_host)
  if(ANDROID)
    set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
    set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER )
    set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER )
  endif()
endmacro()

macro(end_android_find_host)
  if(ANDROID)
    set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM ONLY )
    set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
    set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )
  endif()
endmacro()
