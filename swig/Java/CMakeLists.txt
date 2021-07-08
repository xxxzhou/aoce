include(${SWIG_USE_FILE})
include(AoceHelper)

include_directories("../../code")
include_directories(${ANDROID_NDK}/sources/android/native_app_glue)

set(WRAPPERLIST ../aocewrapper.i)

SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
# 指定包名
set(CMAKE_SWIG_FLAGS -c++ -package aoce.android.library.xswig)
# 输出生成java文件目录
set(AOCE_SWIG_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/android/aoce/src/main/java/aoce/android/library/xswig)
set(AOCE_SWIG_TEMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/swig)
SET(CMAKE_SWIG_OUTDIR ${AOCE_SWIG_TEMP_DIR})
# 生成输出SO目录(和别的C++so文件不在一起,单独指定可能引起编译问题)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/android/aoce/libs/${ANDROID_ABI})

file(GLOB JAVA_TEMP_FILES ${AOCE_SWIG_TEMP_DIR}/*.java)
file(GLOB JAVA_OUTPUT_FILES ${CMAKE_SOURCE_DIR}/*.java)
# force rebuild after CMake run
file(REMOVE ${JAVA_TEMP_FILES} "")
file(REMOVE ${JAVA_OUTPUT_FILES} "")

swig_add_library(aoce_swig_java LANGUAGE java SOURCES ${WRAPPERLIST})
target_link_libraries(aoce_swig_java android aoce aoce_vulkan_extra)

aoce_output(aoce_swig_java) 
install_aoce_module(aoce_swig_java FALSE)

message(STATUS "android swig src: ${AOCE_SWIG_TEMP_DIR}")
message(STATUS "android swig dst: ${AOCE_SWIG_OUTPUT_DIR}")
# 直接复制过去,会导致android studio每次打开把相关包变成目录,使编缉器丢掉包名引用等智能提示出现错误
file(GLOB JAVA_TEMP_FILES ${AOCE_SWIG_TEMP_DIR}/*.java)
if(EXISTS ${AOCE_SWIG_TEMP_DIR})
    message(STATUS "android swig start copy")
    file(COPY ${JAVA_TEMP_FILES}
        DESTINATION ${AOCE_SWIG_OUTPUT_DIR})
endif()
message(STATUS "android swig remove temp file")
file(REMOVE ${JAVA_TEMP_FILES} "")
#add_custom_command(TARGET aoce_swig_java POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy ${AOCE_SWIG_TEMP_DIR} ${AOCE_SWIG_OUTPUT_DIR})



