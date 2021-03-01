include(${SWIG_USE_FILE})
include(AoceHelper)

include_directories("../../code")

set(WRAPPERLIST aocewrapper.i)

SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(${WRAPPERLIST} PROPERTIES CMAKE_SWIG_FLAGS "-includeall")
SET(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_SOURCE_DIR}/Swig" )
SWIG_ADD_MODULE(AoceSwig csharp ${WRAPPERLIST})

target_link_libraries(AoceSwig aoce)

aoce_output(AoceSwig)
install_aoce_module(AoceSwig FLASE) 