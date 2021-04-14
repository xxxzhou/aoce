# GPUImage注解

## 类型说明

GPUImageTwoPassFilter/GPUImageTwoPassTextureSamplingFilter: 说明需要二次PASS,如高斯滤波优化需要执行二次,所以相应类别就从这个类继承.

GPUImageTwoInputFilter: 说明有二个输入文件.

GPUImage3x3ConvolutionFilter/GPUImage3x3TextureSamplingFilter: 给3*3个UV坐标.

GPUImageFilter: 最基本的一输入一输出一pass处理.
