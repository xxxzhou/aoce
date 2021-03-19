# aoce([oeip](https://github.com/xxxzhou/oeip)的android平台扩展版)

android/window 多媒体与游戏引擎交互.

## 配置项目

本项目编写主要使用vscode,配合相应插件C++/CMake。用anroid studio完成anroid平台特定功能与测试等。visual studio 2019配合Nsight调试CUDA相关代码。通过CMake使vscode/android studio/visual studio针对项目文件统一修改。

第三方库引用[aoce_thirdparty](https://github.com/xxxzhou/aoce_thirdparty)
在code平级创建一个目录thirdparty,把aoce_thirdparty里的内容放入这个目录.

vulkan项目使用glsl,请安装Vulkan SDK,通过工具glslangValidator把glsl编译成SPIR-V.

## 做什么

主要想实现一个能在win/andorid方便组合,扩展的GPU图像处理框架.

统一平台win/andorid的视频源的获取,图像的GPU处理,以及方便对接各种界面显示.

GPU计算模块的选择,win平台准备完成cuda/vulkan模块,主要完成cuda,win平台推荐cuda.android平台原则上只实现vulkan模块。

Camera内置WIN平台MF的SDK,而anroid基于ndk camera2实现。

视频的编解码主要基于ffmpeg实现,以及相应推拉流的实现.

cuda/vulkan除了内置的一个简单图像处理,使用者可以以相应cuda/vulkan库为基准,方便自己的layer层实现,其中aoce_talkto/aoce_vulkan_extra分别以aoce_cuda/aoce_vulkan库的类来扩展外置的gpugpu实现,各位可以参照实现。

能方便对接各种引擎,使用各种UI框架进行显示,包含不限于Unity3D/UE4/WinForm等.

各模块现主要通过CMake动态链接,通过相应选项可以选择去除不需要模块,后期完善静态编译成一个文件功能.

## 模块

### [aoce](https://github.com/xxxzhou/aoce)(各个基本功能接口,结构定义)

### aoce_android(android内关于camera/codec相关实现)

### aoce_cuda(aoce图像计算层的cuda实现)

### aoce_ffmpeg(aoce音视频资源播放的ffmpeg实现)

### aoce_vulkan(aoce图像计算层的vulkan实现)

### aoce_win(win平台下基本功能,包含dx11各种资源定义等)

### aoce_win_mf(win平台下aoce图像获取设备的Media Foundation实现)

### aoce_vulkan_extra(用vulkan的compute shader实现gpuimage)
