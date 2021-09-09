include(${SWIG_USE_FILE})
include(AoceHelper)

include_directories("../../code")

set(WRAPPERLIST ../aocewrapper.i)  

SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
# 指定 definition
set(CMAKE_SWIG_FLAGS -O ${SWIG_DEFINITIONS})
SET(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/swig" )

# file(GLOB SHARP_OLD_FILES ${CMAKE_SWIG_OUTDIR}/*.cs)
# file(REMOVE ${SHARP_OLD_FILES} "")  

SWIG_ADD_LIBRARY(aoce_swig LANGUAGE csharp SOURCES ${WRAPPERLIST})

set(SWIG_LIBS aoce)
list(APPEND SWIG_LIBS aoce_vulkan_extra)

if(AOCE_INSTALL_AGORA)
    list(APPEND SWIG_LIBS aoce_talkto_cuda)
endif()

if(AOCE_INSTALL_NCNN)
    list(APPEND SWIG_LIBS aoce_ncnn)
endif()

swig_link_libraries(aoce_swig ${SWIG_LIBS})

aoce_swig_output(aoce_swig)  
install_aoce_swig_module(aoce_swig FALSE)   
message(STATUS "swig: create charp files ${CMAKE_CURRENT_BINARY_DIR}/swig")

# file(GLOB SHARP_FILES "${CMAKE_CURRENT_BINARY_DIR}/swig/*.cs")

# message(STATUS "swig: create charp files ${CMAKE_CURRENT_BINARY_DIR}/swig")
# if(SHARP_FILES)
#     message(STATUS "swig: remove old sharp files")
#     file(GLOB PACKAGE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/swig)
#     # file(REMOVE ${PACKAGE_FILES} "")
#     file(COPY ${SHARP_FILES}
#         DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/swig)    
# endif() 

# ADD_CUSTOM_COMMAND(TARGET aoce_swig
#     POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/swig
# )
