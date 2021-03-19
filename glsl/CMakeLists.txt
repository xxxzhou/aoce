

# 复制glsl文件到bin目录
file(GLOB AOCE_SPV_SOURCE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/target/*.spv" )
message(STATUS "aoce_vulkan_glsl: " ${AOCE_SPV_SOURCE})           
message(STATUS "aoce_vulkan_glsl_dst: " ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/glsl)  
if(WIN32) 
    file(COPY ${AOCE_SPV_SOURCE} DESTINATION "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/glsl")  
    file(COPY ${AOCE_SPV_SOURCE} DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/glsl")  
endif() 