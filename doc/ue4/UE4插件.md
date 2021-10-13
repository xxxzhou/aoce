# UE4插件

UHT(UnrealHeaderTool): 根据定义UField/UFunction等生成元数据,供反射与蓝图等功能使用.

UBT(Unreal Build Tool): 使用C#来完成CMake工程编译,各个模块间的*.Build.cs相当于CMakeLists.txt文件,因此文件夹下有*.Build.cs文件夹的,此文件夹就是一个模块,请记着这个文件夹.

在win平台下,dll文件需要复制到生成的执行目录下,可以使用PublicDelayLoadDLLs表明延迟加载这些dll,但是需要在对应Plugins.cpp中的StartupModule使用FPlatformProcess::GetDllHandle(path)加载指定路径下的dll.
