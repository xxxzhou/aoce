
include(AoceHelper)

set(Realsense_PREFIX_PATH ${AOCE_THIRDPARTY_PATH}/realsense)

create_search_paths(Realsense)

find_path(Realsense_INCLUDE_DIR NAMES "librealsense2/rs.h" HINTS ${Realsense_INC_SEARCH_PATH} PATH_SUFFIXES)

message(STATUS "realsense include: " ${Realsense_INCLUDE_DIR})

find_library(Realsense_LIBRARYS NAME realsense2 HINTS ${Realsense_LIB_SEARCH_PATH} PATH_SUFFIXES)
find_file(Realsense_BINARYS NAME "realsense2.dll" HINTS ${Realsense_BIN_SEARCH_PATH} PATH_SUFFIXES)

message(STATUS "realsense bin: " ${Realsense_LIB_SEARCH_PATH})

if(Realsense_INCLUDE_DIR AND Realsense_LIBRARYS)
    set(Realsense_FOUND TRUE)
    set(Realsense_INCLUDE_DIRS ${Agora_INCLUDE_DIR})
endif()
