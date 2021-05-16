# aoce

## 框架数据流程

数据提供现主要包含如下三种.

1. 摄像头,在win端,有aoce_win_mf模块提供,在android端,有aoce_android提供.

2. 对于多媒体文件(本地多媒体,RTMP等),由aoce_ffmpeg(win/android都支持)提供解码.

3. 直接非压缩的图像二进制数据.

数据处理模块现有aoce_cuda/aoce_vulkan模块处理,win端现支持这二个模块,而android端只支持aoce_vulkan模块.

如果数据提供的是桢数据,对应摄像头/多媒体模块都会解析到VideoFrame并给出回调,而在数据处理模块会有InputLayer层,专门用来接收上面三种数据.

而处理后数据会根据对应OutputLayer需要,导出CPU数据以及GPU数据对接对应系统常用渲染引擎对应纹理上,如在win端,aoce_cuda/aoce_vulkan模块的OutputLayer都支持直接导致到对应DX11纹理,而在android上,aoce_vulkan能直接导致到对应opengl es纹理上,这样就能直接与对应引擎(UE4/Unity3D)底层进行对接.

## 框架导出

拿到源码自己编译,模块内部之间可以随便使用导出类.

如果导出给外部用户使用,需要严格控制只能导出纯净抽像类,或者使用C导出.
