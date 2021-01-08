# aoce([oeip](https://github.com/xxxzhou/oeip)的android平台扩展版)

android/window 多媒体与游戏引擎交互

----

oeip的进阶版本,扩展oeip到android平台,使用cmake编译链接多平台.
第三方库引用[aoce_thirdparty](https://github.com/xxxzhou/aoce_thirdparty)

## 模块

### aoce(各个基本功能接口,结构定义)

1. Layer 各种GPGPU图像计算层接口层
2. Live 直播模块定义层
3. Media 音视频资源播放接口层
4. Module 模块加载与定义
5. VideoDevice 图像获取设备
6. FixGraph 组合一些常用的图像计算层

<!--### aoce_agora(直播模块声网的实现) -->

### aoce_android(android内关于camera/codec相关实现)

### aoce_cuda(aoce图像计算层的cuda实现)

### aoce_ffmpeg(aoce音视频资源播放的ffmpeg实现)

### aoce_vulkan(aoce图像计算层的vulkan实现)

### aoce_win(win平台下基本功能,包含dx11各种资源定义等)

### aoce_win_mf(win平台下aoce图像获取设备的Media Foundation实现)

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

## UE4集成

在UE4Test下查看Plugins/AocePlugins里的相关实现。
