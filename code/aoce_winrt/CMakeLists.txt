include_directories(${AOCE_HEADER_DIR})

# WINRT需要C++ 17
set(CMAKE_CXX_STANDARD 17)

file(GLOB AOCE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )
file(GLOB AOCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_library(aoce_winrt SHARED ${AOCE_SOURCE} ${AOCE_HEADER})

target_precompile_headers(aoce_winrt
	PRIVATE				
		<d3d11.h>
		<DispatcherQueue.h>
		<dwmapi.h>
		<Windows.Graphics.Capture.Interop.h>
		<winrt/Windows.Foundation.Metadata.h>
		<winrt/Windows.Graphics.Capture.h>
		<winrt/Windows.System.h>)

target_link_libraries(aoce_winrt aoce aoce_win Dwmapi windowsapp) 

aoce_output(aoce_winrt)
install_aoce_module(aoce_winrt FLASE)