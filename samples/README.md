# 例子

## 01_findmodule 验证模块功能加载

win端上验证模块加载相关功能.

## 02_showdevice 摄像机模块

win端验证摄像机模块,图像显示需要opencv配置.

## 03_findvulkan vulkan与opencv,vulkan图像处理,opencv显示

win端验证vulkan/cuda计算模块,图像显示需要opencv配置.

## 04_vulkantest (android/win)vulkan自身窗口交换链显示

android端简单演示,win端自身窗口显示,不需要配置opencv.

## 05_livetest (android/win)结合直播模块展示

没有相应模块本demo不能运行.

## [06_mediaplayer](https://zhuanlan.zhihu.com/p/302285687) (android/win)ffmpeg拉流

1. 用ffmpeg打开rtmp流.
2. 使用vulkan Compute shader处理yuv420P/yuv422P数据格式成rgba.
3. 初始化android surface为vulkan的交换链,把如上结果复制到交换链上显示.
4. 如果是opengles surface,如何不通过CPU直接把数据从vulkan复制到opengles里.

需要配置ffmpeg第三方库,android/win各自配置对应的ffmpeg环境,不需要配置opencv.

## 07_androidtest android平台下功能验证

### ndk camera集成

1. ndk camera整合到aoce框架
2. 使用vulkan Compute shader处理yuv420P/yuv422P数据格式成rgba.
3. 初始化android surface为vulkan的交换链,把如上结果复制到交换链上显示.
4. 如果是opengles surface,如何不通过CPU直接把数据从vulkan复制到opengles里.

不需要配置第三方库.

## [07_wintest](https://zhuanlan.zhihu.com/p/349534525) win平台下,cuda/vulkan计算结果与dx11渲染窗口直接交互

使用dx11渲染窗口,不需要配置opencv,需要安装CUDA/DX11.

## 08_vulkanextra gpuimage各种图像效果验证

结合opencv显示vulkanextra各种图像处理,需要配置opencv第三方库.

## [vulkanextratest](https://zhuanlan.zhihu.com/p/348824878) 用于android测试各种vulkanextra里的效果

使用vulkan本身渲染窗口,不需要配置opencv.

注意: 使用Nsight Graphics查看渲染过程,需要本身vulkan的渲染窗口链,如使用opencv,查看不了Nsight Graphics的计算过程.
