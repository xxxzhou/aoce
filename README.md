# aoce([oeip](https://github.com/xxxzhou/oeip)的android平台扩展版)

android/window 多媒体与游戏引擎交互.

## 配置项目

本项目编写主要使用vscode,配合相应插件C++/CMake。用anroid studio完成anroid平台特定功能与测试等。visual studio 2019配合Nsight调试CUDA相关代码。通过CMake使vscode/android studio/visual studio针对项目文件统一修改。

第三方库引用[aoce_thirdparty](https://github.com/xxxzhou/aoce_thirdparty)
在code平级创建一个目录thirdparty,把aoce_thirdparty里的内容放入这个目录.

## 做什么

主要想实现一个能在win/andorid方便组合,扩展的GPU图像处理框架.

统一平台win/andorid的视频源的获取,图像的GPU处理,以及方便对接各种界面显示.

GPU计算模块的选择,win平台准备完成cuda/vulkan/dx11模块,主要完成cuda,win平台推荐cuda.android平台原则上只实现vulkan模块。

Camera内置WIN平台MF的SDK,而anroid基于ndk camera2实现。

视频的编解码主要基于ffmpeg实现,以及相应推拉流的实现.

cuda/vulkan除了内置的一个简单图像处理,使用者可以以相应cuda/vulkan库为基准,方便自己的layer层实现,其中aoce_talkto/aoce_vulkan_extra分别以aoce_cuda/aoce_vulkan库的类来扩展外置的gpugpu实现,各位可以参照实现。

能方便使用各种UI框架进行显示,包含不限于Unity3D/UE4/WinForm等.

各模块现主要通过CMake动态链接,通过相应选项可以选择去除不需要模块,后期完善静态编译成一个文件功能.

## 模块

### aoce(各个基本功能接口,结构定义)

### aoce_android(android内关于camera/codec相关实现)

### aoce_cuda(aoce图像计算层的cuda实现)

### aoce_ffmpeg(aoce音视频资源播放的ffmpeg实现)

### aoce_vulkan(aoce图像计算层的vulkan实现)

### aoce_win(win平台下基本功能,包含dx11各种资源定义等)

### aoce_win_mf(win平台下aoce图像获取设备的Media Foundation实现)

### aoce_vulkan_extra(用vulkan的compute shader实现gpuimage)

## 例子

### 01_findmodule 验证模块功能加载

### 02_showdevice 摄像机模块

### 03_findvulkan vulkan与opencv,vulkan图像处理,opencv显示

### 04_vulkantest (android/win)vulkan自身窗口交换链显示

### 05_livetest (android/win)结合直播模块展示

### [06_mediaplayer](https://zhuanlan.zhihu.com/p/302285687) (android/win)ffmpeg拉流

1. 用ffmpeg打开rtmp流。
2. 使用vulkan Compute shader处理yuv420P/yuv422P数据格式成rgba.
3. 初始化android surface为vulkan的交换链，把如上结果复制到交换链上显示。
4. 如果是opengles surface,如何不通过CPU直接把数据从vulkan复制到opengles里。

### 07_androidtest android平台下功能验证

#### ndk camera集成

    1. ndk camera整合到aoce框架
    2. 使用vulkan Compute shader处理yuv420P/yuv422P数据格式成rgba.
    3. 初始化android surface为vulkan的交换链，把如上结果复制到交换链上显示。
    4. 如果是opengles surface,如何不通过CPU直接把数据从vulkan复制到opengles里。

### 07_wintest win平台下,cuda计算结果与dx11渲染窗口直接交互

### 08_vulkanextra gpuimage各种图像效果验证

### vulkanextratest 用于android测试各种vulkanextra里的效果

## UE4集成

在UE4Test下查看Plugins/AocePlugins里的相关实现。
