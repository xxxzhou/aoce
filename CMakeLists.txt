cmake_minimum_required(VERSION 3.4.2)

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

project(aoce)

# include函数的路径
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
# 查找第三方库
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/thirdparty")

# 设置本项目的安装目录
include(GNUInstallDirs)
# 如果CMAKE_INSTALL_PREFIX没设置值,用于默认初始化 CMAKE_BINARY_DIR为build目录 CACHE序列化
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND WIN32)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE)
endif()
message(STATUS "install:" ${CMAKE_INSTALL_PREFIX})

# 添加win32平台下的编译符号
if(WIN32)
    # Treat warnings as errors
    #add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/WX>")
    # Disable RTTI
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/GR->")
    # Warn about nested declarations
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34456>")
    # Warn about potentially uninitialized variables
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34701>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34703>")
    # Warn about different indirection types.
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34057>")
    # Warn about signed/unsigned mismatch.
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/w34245>")
endif()

message(STATUS "build type:" ${CMAKE_BUILD_TYPE})
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "choose the type of build" FORCE)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG) 
endif ()

set(AOCE_HEADER_DIR "${CMAKE_SOURCE_DIR}/code/aoce")
set(AOCE_VULKAN_HEADER_DIR "${CMAKE_SOURCE_DIR}/code/aoce_vulkan")

# 设置build的子目录
set(AOCE_RUNTIME_DIR ${CMAKE_BINARY_DIR}/bin)
set(AOCE_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)

# 生成目录
function(aoce_output targetname) 
    # message(STATUS "output" ${targetname})
    set_target_properties(${targetname} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${AOCE_RUNTIME_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${AOCE_LIBRARY_DIR}     
    ARCHIVE_OUTPUT_DIRECTORY ${AOCE_LIBRARY_DIR}
    )
endfunction(aoce_output targetname)

add_subdirectory(code)
add_subdirectory(samples)


