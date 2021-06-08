include(${SWIG_USE_FILE})
include(AoceHelper)

include_directories("../../code")

set(WRAPPERLIST ../aocewrapper.i) 

SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
SET(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/swig" )

swig_add_library(aoce_swig LANGUAGE csharp SOURCES ${WRAPPERLIST})
target_link_libraries(aoce_swig aoce aoce_vulkan_extra)

aoce_swig_output(aoce_swig) 
install_aoce_swig_module(aoce_swig FALSE) 
message(STATUS "swig: create charp files ${CMAKE_CURRENT_BINARY_DIR}/swig")
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/swig)

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
