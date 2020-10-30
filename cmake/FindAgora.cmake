
include(AoceHelper)

start_android_find_host()

set(Agora_PREFIX_PATH ${AOCE_THIRDPARTY_PATH}/agora)

create_search_paths(Agora)

find_path(Agora_INCLUDE_DIR NAME AgoraBase.h HINTS ${Agora_INC_SEARCH_PATH} PATH_SUFFIXES)

message(STATUS "agora include:" ${Agora_INC_SEARCH_PATH})

if(WIN32)
    find_library(Agora_LIBRARYS NAME agora_rtc_sdk HINTS ${Agora_LIB_SEARCH_PATH} PATH_SUFFIXES)
    find_file(Agora_BINARYS NAME "agora_rtc_sdk.dll" HINTS ${Agora_BIN_SEARCH_PATH} PATH_SUFFIXES)
elseif(ANDROID)
    find_library(Agora_LIBRARYS NAME agora-rtc-sdk-jni HINTS ${Agora_LIB_SEARCH_PATH} PATH_SUFFIXES)
endif()

message(STATUS "agora bin " ${Agora_BIN_SEARCH_PATH})

if(Agora_INCLUDE_DIR AND Agora_LIBRARYS)
    set(Agora_FOUND TRUE)
    set(Agora_INCLUDE_DIRS ${Agora_INCLUDE_DIR})
endif()

end_android_find_host()

