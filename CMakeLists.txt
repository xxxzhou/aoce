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

# 添加win32平台下的编译符号
if(WIN32)
    set(CMAKE_CXX_STANDARD 14)
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

set(AOCE_SET_STATIC 0)
option(AOCE_INSTALL_FFMPEG "install ffmpeg module" TRUE)
cmake_dependent_option(AOCE_INSTALL_AGORA "install agora module" TRUE "AOCE_AGORA_EXIST" FALSE) #FALSE
cmake_dependent_option(AOCE_INSTALL_TALKTO "install talkto module" TRUE "WIN32;AOCE_TALKTO_EXIST" FALSE)

# 配置联系cmake/c++
configure_file(${CMAKE_SOURCE_DIR}/cmake/AoceBuildSettings.h.in ${CMAKE_SOURCE_DIR}/code/aoce/AoceBuildSettings.h @ONLY)
install(FILES ${CMAKE_SOURCE_DIR}/code/aoce/AoceBuildSettings.h DESTINATION include/aoce)

add_subdirectory(code)
add_subdirectory(samples)

find_package(SWIG 4.0)
if(SWIG_FOUND AND WIN32)
    message(STATUS "SWIG found: ${SWIG_EXECUTABLE}")
    add_subdirectory(CSharp)
endif()




