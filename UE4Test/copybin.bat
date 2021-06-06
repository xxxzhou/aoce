@echo off
echo copy window/android install to ue4 aoce plugins
echo=
echo====copy mode====
	echo = 1 debug(pdb)
	echo = 2 install
echo================
set/p copyMode=
if %copyMode%== 1 (
	echo start copy debug
) ^
else if %copyMode%== 2 (
	echo start copy install
)
set curdir=%~dp0
cd /d %curdir%

timeout 1
echo start copy include
xcopy /s /e /y /c /i "..\build\install\win\include" ".\Plugins\AocePlugins\ThirdParty\Aoce\win\include"
echo end copy  include

timeout 1
echo start copy window bin
if %copyMode%== 1 (
	xcopy /s /e /y /c /i "..\build\bin\Debug" ".\Plugins\AocePlugins\ThirdParty\Aoce\win\bin"
	xcopy /s /e /y /c /i "..\build\lib\Debug" ".\Plugins\AocePlugins\ThirdParty\Aoce\win\lib"
) ^
else if %copyMode%== 2 (
	xcopy /s /e /y /c /i "..\build\install\win\bin" ".\Plugins\AocePlugins\ThirdParty\Aoce\win\bin"
	xcopy /s /e /y /c /i "..\build\install\win\lib" ".\Plugins\AocePlugins\ThirdParty\Aoce\win\lib"
)
xcopy /s /e /y /c /i "..\glsl\target" ".\Plugins\AocePlugins\ThirdParty\Aoce\win\bin\glsl"
copy "..\thirdparty\agora\x64\dll\agora_rtc_sdk.dll" ".\Plugins\AocePlugins\ThirdParty\Aoce\win\bin\agora_rtc_sdk.dll"
copy "..\thirdparty\realsense\x64\dll\realsense2.dll" ".\Plugins\AocePlugins\ThirdParty\Aoce\win\bin\realsense2.dll"
echo end copy bin

timeout 1
echo start copy android bin
set androidDir=05_livetest
:: armeabi-v7a arm64-v8a
set androidAbi=arm64-v8a
if %copyMode%== 1 (
	xcopy /s /e /y /c /i "..\android\%androidDir%\build\intermediates\cmake\debug\obj\armeabi-v7a" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\armeabi-v7a"
	xcopy /s /e /y /c /i "..\android\%androidDir%\build\intermediates\cmake\debug\obj\arm64-v8a" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\arm64-v8a"	
) ^
else if %copyMode%== 2 (
	xcopy /s /e /y /c /i "..\android\%androidDir%\build\intermediates\cmake\release\obj\armeabi-v7a" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\armeabi-v7a"
	xcopy /s /e /y /c /i "..\android\%androidDir%\build\intermediates\cmake\release\obj\arm64-v8a" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\arm64-v8a"	
)
xcopy /s /e /y /c /i "..\glsl\target" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\glsl"
copy "..\thirdparty\agora\android\agora-rtc-sdk.jar" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\agora-rtc-sdk.jar"
copy "..\thirdparty\agora\android\armeabi-v7a\libagora-rtc-sdk-jni.so" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\armeabi-v7a\libagora-rtc-sdk-jni.so"
copy "..\thirdparty\agora\android\armeabi-v7a\libagora-ffmpeg.so" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\armeabi-v7a\libagora-ffmpeg.so"
copy "..\thirdparty\agora\android\arm64-v8a\libagora-rtc-sdk-jni.so" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\arm64-v8a\libagora-rtc-sdk-jni.so"
copy "..\thirdparty\agora\android\arm64-v8a\libagora-ffmpeg.so" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\arm64-v8a\libagora-ffmpeg.so"
echo end copy android

timeout 1
echo start copy ffmpeg
set copyffmpeg = 1
:if %copyffmpeg%== 0
xcopy /s /e /y /c /i "..\thirdparty\ffmpeg\x64\dll" ".\Plugins\AocePlugins\ThirdParty\Aoce\win\bin"
xcopy /s /e /y /c /i "..\thirdparty\ffmpeg\android\armeabi-v7a" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\armeabi-v7a"
xcopy /s /e /y /c /i "..\thirdparty\ffmpeg\android\arm64-v8a" ".\Plugins\AocePlugins\ThirdParty\Aoce\android\arm64-v8a"
echo end copy ffmpeg

