cmake_minimum_required(VERSION 3.4.2)

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

project(aoce)

# include函数的路径
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
# 查找第三方库
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/thirdparty")
set(AOCE_THIRDPARTY_PATH ${CMAKE_PREFIX_PATH})

# cmake option(cmake_dependent_option)
include(CMakeDependentOption) 

# 设置本项目的安装目录
include(GNUInstallDirs)
# 如果CMAKE_INSTALL_PREFIX没设置值,用于默认初始化 CMAKE_BINARY_DIR为build目录 CACHE序列化
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND WIN32)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install/win" CACHE PATH "default install path" FORCE)
elseif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND ANDROID)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install/android" CACHE PATH "default install path" FORCE)
endif()
message(STATUS "install:" ${CMAKE_INSTALL_PREFIX})

if(ANDROID)
    message(STATUS "android abi:" ${ANDROID_ABI})
    message(STATUS "android ndk:" ${ANDROID_NDK})
endif()
# 检查32bit or 64bit system
include(CheckTypeSize)
CHECK_TYPE_SIZE("void*" AOCE_PTR_SIZE BUILTIN_TYPES_ONLY)
if (AOCE_PTR_SIZE EQUAL 8)
    set(AOCE_PLATFORM_X64 TRUE)
else ()
    set(AOCE_PLATFORM_X64 FALSE)
endif ()

# CLANG加上debug编译符
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DAOCE_DEBUG_TYPE=1" )
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DAOCE_DEBUG_TYPE=1" )
endif()

# 添加win32平台下的编译符号
if(WIN32)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    # 下面这些选项会导致aoce_cuda里nvcc编译不通过
    # # Treat warnings as errors
    # #add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/WX>")
    # # Disable RTTI
    # add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/GR->")
    # # Warn about nested declarations
    # add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34456>")
    # # Warn about potentially uninitialized variables
    # add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34701>")
    # add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34703>")
    # # Warn about different indirection types.
    # add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34057>")
    # # Warn about signed/unsigned mismatch.
    # add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34245>")
endif()
if(ANDROID)
    set(CMAKE_CXX_VISIBILITY_PRESET default)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)
endif()
# variable_watch(CMAKE_BUILD_TYPE)
# set(CMAKE_BUILD_TYPE "Release"  CACHE STRING "choose the type of build" FORCE)
message(STATUS "build type:" ${CMAKE_BUILD_TYPE})   
if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "choose the type of build" FORCE)
endif() 
set(AOCE_DEBUG_TYPE 0) 
if (CMAKE_BUILD_TYPE STREQUAL "Debug") 
    set(AOCE_DEBUG_TYPE 1)
endif ()

set(AOCE_HEADER_DIR "${CMAKE_SOURCE_DIR}/code/aoce")
set(AOCE_VULKAN_HEADER_DIR "${CMAKE_SOURCE_DIR}/code/aoce_vulkan")

# 设置build的子目录
set(AOCE_RUNTIME_DIR ${CMAKE_BINARY_DIR}/bin)
set(AOCE_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)

# 检查是否有aoce_agora实现
set(AOCE_AGORA_EXIST FALSE)
if(EXISTS "${CMAKE_SOURCE_DIR}/code/aoce_agora")
    message(STATUS "find aoce_agora")
    set(AOCE_AGORA_EXIST TRUE)
endif()
# 检查是否有aoce_talkto实现
set(AOCE_TALKTO_EXIST FALSE)
if(EXISTS "${CMAKE_SOURCE_DIR}/code/aoce_talkto")
    message(STATUS "find aoce_talkto")
    set(AOCE_TALKTO_EXIST TRUE)
endif()
# 检查是否有aoce_unity3d实现
set(AOCE_UNITY3D_EXIST FALSE)
if(EXISTS "${CMAKE_SOURCE_DIR}/code/aoce_unity3d")
    message(STATUS "find aoce_unity3d")
    set(AOCE_UNITY3D_EXIST TRUE)
endif()

set(AOCE_SET_STATIC 0)
option(AOCE_INSTALL_FFMPEG "install ffmpeg module" OFF)
option(AOCE_ENABLE_SAMPLES "enable samples" TRUE) 
option(AOCE_ENABLE_WINRT "enable winrt" OFF) 
option(AOCE_ENABLE_SWIG "enable swig" TRUE) 
option(AOCE_INSTALL_NCNN "install ncnn module" TRUE)

cmake_dependent_option(AOCE_CUDA_MODULE "install cuda module" TRUE "WIN32" FALSE)
cmake_dependent_option(AOCE_INSTALL_AGORA "install agora module" TRUE "AOCE_AGORA_EXIST" FALSE) #FALSE

# 配置联系cmake/c++
configure_file(${CMAKE_SOURCE_DIR}/cmake/AoceBuildSettings.h.in ${CMAKE_SOURCE_DIR}/code/aoce/AoceBuildSettings.h @ONLY)
# 导出给外部用户使用的C头文件,主要是抽像类/结构/C风格创建对象三种
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceBuildSettings.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceDefine.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/Aoce.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceCore.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceLayer.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceLive.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceMath.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceMedia.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceVideoDevice.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceMetadata.h DESTINATION include/aoce) 
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceWindow.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceAudioDevice.h DESTINATION include/aoce)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce_vulkan_extra/AoceVkExtra.h DESTINATION include/aoce_vulkan_extra)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce_vulkan_extra/VkExtraExport.h DESTINATION include/aoce_vulkan_extra) 
if(AOCE_TALKTO_EXIST)
    install(FILES ${CMAKE_SOURCE_DIR}/code/aoce_talkto/TalkTo.h DESTINATION include/aoce_talkto)
endif()

add_subdirectory(code)
add_subdirectory(samples)
add_subdirectory(glsl)

if(AOCE_ENABLE_SWIG) 
    # REQUIRED 
    message(STATUS "start swig convert other language")
    find_package(SWIG)
    if(SWIG_FOUND)       
        add_subdirectory(swig)
    endif(SWIG_FOUND)
endif()




