include(AoceHelper)
start_android_find_host()

set(FFmpeg_PREFIX_PATH ${AOCE_THIRDPARTY_PATH}/ffmpeg)
create_search_paths(FFmpeg)

message(STATUS "ffmpeg include:" ${FFmpeg_INC_SEARCH_PATH})
message(STATUS "ffmpeg libs:" ${FFmpeg_LIB_SEARCH_PATH})

set(FFMPEG_INCLUDE_DIRS)
set(FFMPEG_LIBRARIES)
set(FFMPEG_BINARYS)

macro(ffmepg_find_component component header)
    string(TOUPPER "${component}" UCOMPONENT)
    set(FFMPEG_${UCOMPONENT}_FOUND FLASE)
    set(FFmpeg_${component}_FOUND FLASE)
    find_path(FFMPEG_${component}_INCLUDE_DIR NAMES "lib${component}/${header}" HINTS ${FFmpeg_INC_SEARCH_PATH} PATH_SUFFIXES)
    find_library(FFMPEG_${component}_LIBRARY NAMES "${component}" "lib${component}" HINTS ${FFmpeg_LIB_SEARCH_PATH} PATH_SUFFIXES)
    set(FFMPEG_${UCOMPONENT}_INCLUDE_DIRS ${FFMPEG_${component}_INCLUDE_DIR} )
    set(FFMPEG_${UCOMPONENT}_LIBRARIES ${FFMPEG_${component}_LIBRARY} )

    message(STATUS "ffmpeg ${component} include:" ${FFMPEG_${component}_INCLUDE_DIR})
    message(STATUS "ffmpeg ${component} libs:" ${FFMPEG_${component}_LIBRARY})
    # https://cmake.org/cmake/help/v3.0/command/if.html
    # if直接填写变量 
    if(FFMPEG_${component}_INCLUDE_DIR AND FFMPEG_${component}_LIBRARY)
        set(FFMPEG_${UCOMPONENT}_FOUND TRUE)
        set(FFmpeg_${component}_FOUND TRUE)        

        # 添加到FFMPEG_INCLUDE_DIRS,然后去重
        list(APPEND FFMPEG_INCLUDE_DIRS ${FFMPEG_${component}_INCLUDE_DIR})
        list(REMOVE_DUPLICATES FFMPEG_INCLUDE_DIRS)
        set(FFMPEG_INCLUDE_DIRS "${FFMPEG_INCLUDE_DIRS}" PARENT_SCOPE)

        list(APPEND FFMPEG_LIBRARIES ${FFMPEG_${component}_LIBRARY})
		list(REMOVE_DUPLICATES FFMPEG_LIBRARIES)
		set(FFMPEG_LIBRARIES "${FFMPEG_LIBRARIES}" PARENT_SCOPE)

        set(FFMPEG_${UCOMPONENT}_VERSION_STRING "unknown")
        set(_vfile "${FFMPEG_${component}_INCLUDE_DIR}/lib${component}/version.h")        
        if(EXISTS "${_vfile}")
            file(STRINGS "${_vfile}" _version_parse REGEX "^.*VERSION_(MAJOR|MINOR|MICRO)[ \t]+[0-9]+[ \t]*$")
            string(REGEX REPLACE ".*VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" _major "${_version_parse}")
            string(REGEX REPLACE ".*VERSION_MINOR[ \t]+([0-9]+).*" "\\1" _minor "${_version_parse}")
            string(REGEX REPLACE ".*VERSION_MICRO[ \t]+([0-9]+).*" "\\1" _micro "${_version_parse}")

            set(FFMPEG_${UCOMPONENT}_VERSION_MAJOR "${_major}")
            set(FFMPEG_${UCOMPONENT}_VERSION_MINOR "${_minor}")
            set(FFMPEG_${UCOMPONENT}_VERSION_MICRO "${_micro}")

            set(FFMPEG_${UCOMPONENT}_VERSION_STRING "${_major}.${_minor}.${_micro}")           

            if(WIN32)                  
                find_file(FFMPEG_${component}_BINARYS NAME "${component}-${_major}.dll" HINTS ${FFmpeg_BIN_SEARCH_PATH} PATH_SUFFIXES)
                list(APPEND FFMPEG_BINARYS ${FFMPEG_${component}_BINARYS})
                set(FFMPEG_BINARYS "${FFMPEG_BINARYS}" PARENT_SCOPE)                
            endif()
        else()
            message(STATUS "Failed parsing FFmpeg ${component} version")
        endif()
    endif()
endmacro()

if(WIN32)
    ffmepg_find_component("avcodec" "avcodec.h")
    ffmepg_find_component("swresample" "swresample.h")
    ffmepg_find_component("avutil" "avutil.h")
    ffmepg_find_component("avformat" "avformat.h")
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(FFmpeg
        FOUND_VAR FFMPEG_FOUND
        REQUIRED_VARS FFMPEG_AVCODEC_LIBRARIES FFMPEG_AVCODEC_INCLUDE_DIRS
        VERSION_VAR FFMPEG_AVCODEC_VERSION_STRING
        HANDLE_COMPONENTS)
elseif(ANDROID)
    find_path(FFMPEG_INCLUDE_DIRS NAME libavcodec/avcodec.h HINTS ${FFmpeg_INC_SEARCH_PATH} PATH_SUFFIXES)
    find_library(FFMPEG_LIBRARIES NAME ffmpeg HINTS ${FFmpeg_LIB_SEARCH_PATH} PATH_SUFFIXES ${ANDROID_ABI})
    message(STATUS "android ffmpeg inc:" ${FFMPEG_INCLUDE_DIRS})
    message(STATUS "android ffmpeg lib:" ${FFMPEG_LIBRARIES})
    if(FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARIES)
        set(FFmpeg_FOUND TRUE)        
    endif()
endif()
    
end_android_find_host()