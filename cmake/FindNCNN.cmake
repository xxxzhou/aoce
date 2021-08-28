include(AoceHelper)

start_android_find_host()

set(NCNN_PREFIX_PATH ${AOCE_THIRDPARTY_PATH}/ncnn)

create_search_paths(NCNN)

find_path(NCNN_INCLUDE_DIR NAME net.h HINTS ${NCNN_INC_SEARCH_PATH} PATH_SUFFIXES)

message(STATUS "ncnn include:" ${NCNN_INCLUDE_DIR})

if(WIN32)    
    if (AOCE_DEBUG_TYPE)         
        find_library(NCNN_LIBRARYS NAME ncnnd HINTS ${NCNN_LIB_SEARCH_PATH} PATH_SUFFIXES)
        find_file(NCNN_BINARYS NAME "ncnnd.dll" HINTS ${NCNN_BIN_SEARCH_PATH} PATH_SUFFIXES)
    else()
        find_library(NCNN_LIBRARYS NAME ncnn HINTS ${NCNN_LIB_SEARCH_PATH} PATH_SUFFIXES)
        find_file(NCNN_BINARYS NAME "ncnn.dll" HINTS ${NCNN_BIN_SEARCH_PATH} PATH_SUFFIXES)
    endif()
elseif(ANDROID)
    find_library(NCNN_LIBRARYS NAME ncnn HINTS ${NCNN_LIB_SEARCH_PATH} PATH_SUFFIXES ${ANDROID_ABI})
endif()

if(NCNN_INCLUDE_DIR AND NCNN_LIBRARYS)
    set(NCNN_FOUND TRUE)
    set(NCNN_INCLUDE_DIRS ${NCNN_INCLUDE_DIR})
endif()

end_android_find_host()