
set (SWIG_DEFINITIONS "")

function(add_swig_definitions def)   
    if("${SWIG_DEFINITIONS}" STREQUAL "")
        string(APPEND SWIG_DEFINITIONS "${def}")
    else()
        string(APPEND SWIG_DEFINITIONS " " "${def}")
    endif() 
    set(SWIG_DEFINITIONS ${SWIG_DEFINITIONS} PARENT_SCOPE)
endfunction(add_swig_definitions def)

if (AOCE_INSTALL_AGORA)
    add_swig_definitions("-DAOCE_INSTALL_AGORA")
endif (AOCE_INSTALL_AGORA) 

if (AOCE_INSTALL_NCNN)
    add_swig_definitions("-DAOCE_INSTALL_NCNN")
endif (AOCE_INSTALL_NCNN)

if(WIN32)
    add_swig_definitions("-DWIN32")    
elseif(ANDROID)
    # add_swig_definitions("-D__ANDROID__")    
endif()  

message(STATUS "swig definitons:" ${SWIG_DEFINITIONS})

if(WIN32)    
    add_subdirectory(CSharp)
elseif(ANDROID)    
    add_subdirectory(Java)
endif()


