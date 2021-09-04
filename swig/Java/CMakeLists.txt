include(${SWIG_USE_FILE})
include(AoceHelper)

start_android_find_host()

set(WRAPPERLIST ../aocewrapper.i)
set(AOCE_JAVA_PACKAGE "aoce.android.library.xswig")
string(REPLACE "." "/" AOCE_JAVA_SUBDIR ${AOCE_JAVA_PACKAGE})
# 输出生成java文件目录
set(AOCE_SWIG_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/android/aoce/src/main/java/${AOCE_JAVA_SUBDIR})
set(AOCE_SWIG_TEMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/swig)

include_directories("../../code")
include_directories(${ANDROID_NDK}/sources/android/native_app_glue)

SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CPLUSPLUS ON)
# SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
# 指定包名
set(CMAKE_SWIG_FLAGS -c++ -package aoce.android.library.xswig -O ${SWIG_DEFINITIONS})
SET(CMAKE_SWIG_OUTDIR ${AOCE_SWIG_TEMP_DIR})

# 生成输出SO目录(和别的C++so文件不在一起,单独指定可能引起编译问题)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/android/aoce/libs/${ANDROID_ABI})
SWIG_ADD_LIBRARY(aoce_swig_java LANGUAGE java SOURCES ${WRAPPERLIST})
if(AOCE_INSTALL_AGORA)
    swig_link_libraries(aoce_swig_java android aoce aoce_vulkan_extra aoce_talkto aoce_ncnn)
else()
    swig_link_libraries(aoce_swig_java android aoce aoce_vulkan_extra aoce_ncnn)
endif()

# 直接复制过去,会导致android studio每次打开把相关包变成目录,使编缉器丢掉包名引用等智能提示出现错误
file(GLOB JAVA_NTEMP_FILES ${AOCE_SWIG_TEMP_DIR}/*.java)
file(GLOB JAVA_NOW_FILES ${AOCE_SWIG_OUTPUT_DIR}/*.java)
message(STATUS "android swig src dir: ${AOCE_SWIG_TEMP_DIR}")
message(STATUS "android swig dst dir: ${AOCE_SWIG_OUTPUT_DIR}")
#if(JAVA_NTEMP_FILES)
    message(STATUS "android swig start copy")
    # 移除老的生成文件
    #file(REMOVE ${JAVA_NOW_FILES} "")
    # copy新的生成文件
    file(COPY ${JAVA_NTEMP_FILES}
        DESTINATION ${AOCE_SWIG_OUTPUT_DIR})
    file(REMOVE ${JAVA_NTEMP_FILES} "")
#endif()

end_android_find_host()





