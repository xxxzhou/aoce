# aoce

android/window 多媒体与游戏交互

## [06_mediaplayer](https://zhuanlan.zhihu.com/p/302285687)

1. 用ffmpeg打开rtmp流。
2. 使用vulkan Compute shader处理yuv420P/yuv422P数据格式成rgba.
3. 初始化android surface为vulkan的交换链，把如上结果复制到交换链上显示。
4. 如果是opengles surface,如何不通过CPU直接把数据从vulkan复制到opengles里。
