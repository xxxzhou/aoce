function(add_swig_definitions def)   
    if("${SWIG_DEFINITIONS}" STREQUAL "")
        string(APPEND SWIG_DEFINITIONS "${def}")
    else()
        string(APPEND SWIG_DEFINITIONS " " "${def}")
    endif() 
    set(SWIG_DEFINITIONS ${SWIG_DEFINITIONS} PARENT_SCOPE)
endfunction(add_swig_definitions def)

if (AOCE_INSTALL_AGORA)  
    # 对应%{...%}代码段里宏定义
    add_definitions(-DAOCE_INSTALL_AGORA)    
    # 对应swig语法中的宏定义
    list(APPEND SWIG_DEFINITIONS -DAOCE_INSTALL_AGORA)
endif() 

if (AOCE_INSTALL_NCNN)  
    add_definitions(-DAOCE_INSTALL_NCNN) 
    list(APPEND SWIG_DEFINITIONS -DAOCE_INSTALL_NCNN)
endif()

message(STATUS "swig definitons:" ${SWIG_DEFINITIONS}) 

if(WIN32)    
    add_subdirectory(CSharp)
elseif(ANDROID)     
    add_subdirectory(Java)
endif()


