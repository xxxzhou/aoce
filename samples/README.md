# 例子

## 01_findmodule 验证模块功能加载

## 02_showdevice 摄像机模块

## 03_findvulkan vulkan与opencv,vulkan图像处理,opencv显示

## 04_vulkantest (android/win)vulkan自身窗口交换链显示

## 05_livetest (android/win)结合直播模块展示

## [06_mediaplayer](https://zhuanlan.zhihu.com/p/302285687) (android/win)ffmpeg拉流

1. 用ffmpeg打开rtmp流。
2. 使用vulkan Compute shader处理yuv420P/yuv422P数据格式成rgba.
3. 初始化android surface为vulkan的交换链，把如上结果复制到交换链上显示。
4. 如果是opengles surface,如何不通过CPU直接把数据从vulkan复制到opengles里。

## 07_androidtest android平台下功能验证

### ndk camera集成

    1. ndk camera整合到aoce框架
    2. 使用vulkan Compute shader处理yuv420P/yuv422P数据格式成rgba.
    3. 初始化android surface为vulkan的交换链，把如上结果复制到交换链上显示。
    4. 如果是opengles surface,如何不通过CPU直接把数据从vulkan复制到opengles里。

## [07_wintest](https://zhuanlan.zhihu.com/p/349534525) win平台下,cuda/vulkan计算结果与dx11渲染窗口直接交互

## 08_vulkanextra gpuimage各种图像效果验证

## [vulkanextratest](https://zhuanlan.zhihu.com/p/348824878) 用于android测试各种vulkanextra里的效果