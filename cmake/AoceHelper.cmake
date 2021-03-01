
# 添加子文件夹里文件并归类
function(add_sub_path relativePath HEADER_FILES SOURCE_FILELIST)
    string(REPLACE "/" "\\" filterPart ${relativePath})
    file(GLOB TEMP_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${relativePath}/*.cpp") 
    file(GLOB TEMP_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/${relativePath}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/${relativePath}/*.hpp") 
    # 这是列表的操作方式    
    set(${HEADER_FILES} ${${HEADER_FILES}} ${TEMP_HEADER} PARENT_SCOPE)
    set(${SOURCE_FILELIST} ${${SOURCE_FILELIST}} ${TEMP_SOURCE} PARENT_SCOPE)
    source_group(${filterPart} FILES ${TEMP_HEADER} ${TEMP_SOURCE})  
endfunction()

# 生成目录
function(aoce_output targetname) 
    # message(STATUS "output" ${targetname})
    set_target_properties(${targetname} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${AOCE_RUNTIME_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${AOCE_LIBRARY_DIR}     
    ARCHIVE_OUTPUT_DIRECTORY ${AOCE_LIBRARY_DIR}
    )
endfunction(aoce_output targetname)

# 输出文件到install目录
function(install_aoce_module module contain_include) 
  install(TARGETS ${module}
          EXPORT ${module}EXPORT
          # CONFIGURATIONS Release
          LIBRARY DESTINATION lib  # 动态库安装路径
          ARCHIVE DESTINATION lib  # 静态库安装路径
          RUNTIME DESTINATION bin  # 可执行文件安装路径
          # PUBLIC_HEADER DESTINATION include/${module}  # 头文件安装路径
          )   
   # 复制头文件        
  if(contain_include)      
    message(STATUS "copy ${module} include") 
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include 
            FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
            )
  endif()      
endfunction(install_aoce_module)


# Construct search paths for includes and libraries from a PREFIX_PATH
macro(create_search_paths PREFIX)
  foreach(dir ${${PREFIX}_PREFIX_PATH})
    if(WIN32)
      set(platform_dir ${dir}/x64)
    elseif(ANDROID)
      set(platform_dir ${dir}/android)
    endif()    
    set(${PREFIX}_INC_SEARCH_PATH ${${PREFIX}_INC_SEARCH_PATH}
      ${platform_dir}/include ${platform_dir}/Include ${platform_dir}/include/${PREFIX} ${platform_dir}/Headers)
    set(${PREFIX}_INC_SEARCH_PATH ${${PREFIX}_INC_SEARCH_PATH}
      ${dir}/include ${dir}/Include ${dir}/include/${PREFIX} ${dir}/Headers)
    set(${PREFIX}_LIB_SEARCH_PATH ${${PREFIX}_LIB_SEARCH_PATH} ${platform_dir}
      ${platform_dir}/lib ${platform_dir}/Lib ${platform_dir}/lib/${PREFIX} ${platform_dir}/Libs)
    set(${PREFIX}_BIN_SEARCH_PATH ${${PREFIX}_BIN_SEARCH_PATH} ${platform_dir}/dll ${platform_dir}/bin )
  endforeach(dir)
endmacro(create_search_paths)

# window环境下编译android
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
