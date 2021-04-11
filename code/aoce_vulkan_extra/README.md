# 移植GPUImage的实现,用vulkan的compute shader完成各个功能

[iOS直播技术学习笔记-美颜滤镜效果（三）](https://www.jianshu.com/p/90f55e5b7d16)
[GPUImage 简介](https://gitee.com/xudoubi/GPUImage)

## opengl/cuda computer shader 线程

gl_NumWorkGroups/gridDim: 所有线程块的多少.
gl_WorkGroupSize/blockDim: 本身线程块的大小.
gl_WorkGroupID/blockIdx: 线程块在所有线程块中索引.
gl_LocalInvocationID/threadIdx: 线程在线程块中的索引.
gl_GlobalInvocationID = blockIdx*blockDim + gl_LocalInvocationID

## 功能介绍

### ChromKey

[UE4 Matting](https://www.unrealengine.com/en-US/tech-blog/setting-up-a-chroma-key-material-in-ue4)

主要注意一点,UBO,我特意把一个float,vec3放一起,想当然的认为是按照vec4排列,这里注意,vec3不管前后接什么,按照vec4排的.

### Luminance 取亮度

各种标准，其中人眼对绿色比较敏感，绿色占比最高，红色次之。

### BoxBlur 均值模糊

[图像处理中的卷积核分离](https://zhuanlan.zhihu.com/p/81683945)

可以从上图推导过程中看出，一个m行乘以n列的高斯卷积可以分解成一个1行乘以n列的行卷积，之后串联一个m行乘以1列的列卷积的形式，输出保持不变。行卷积的卷积核参数（均值和方差）等于原始m行n列卷积核在列方向（Y方向）的均值和方差，列卷积的卷积核参数等于原始m行n列卷积核在行方向（X方向）上的均值和方差。

opencv cudafilters的高斯卷积,就采用先计算水平卷积,然后计算垂直卷积的优化方式.

其中边框模式暂时先固定为REPLICATE.

[调整图像边缘](https://blog.csdn.net/shuiyixin/article/details/106472722)

![Alt text](https://img-blog.csdnimg.cn/20200602093115149.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NodWl5aXhpbg==,size_16,color_47FFFF,t_70 "REPLICATE image")

### GaussianBlur

[Markdown 数学公式语法](https://www.jianshu.com/p/4460692eece4)

[Cmd Markdown 公式指导手册](https://www.zybuluo.com/codeep/note/163962)

[OpenCV高斯滤波GaussianBlur](https://blog.csdn.net/godadream/article/details/81568844)

在android 1080P下20*20的核，Radmi K10 Pro不到一桢.

优化方向

1 卷积核分离，一个m行乘以n列的高斯卷积可以分解成一个1行乘以n列的行卷积，常用的BOX与高斯都可以分解。

2 利用shared局部显存减少访问纹理显存的操作,注意这块容量非常有限,如果不合理分配，能并行的组就少了.考虑到Android平台，使用packUnorm4x8/unpackUnorm4x8优化局部显存占用。

主要使用opencv里opencv_cudafilters模块代码。

|  |row kenrel| |
| ------ | ------ | ------ |
| WorkGroupSize*HALO_SIZE | WorkGroupSize*PATCH_PER_BLOCK | WorkGroupSize*HALO_SIZE |

| column kenrel |
| --- |
|WorkGroupSize*HALO_SIZE|
|WorkGroupSize*PATCH_PER_BLOCK|
|WorkGroupSize*HALO_SIZE|

其中正常PATCH_PER_BLOCK是一个线程操作几个像素，一般常用图像处理操作来说，我们都是1：1，在这设置4。
针对图像块就是WorkGroupSize*PATCH_PER_BLOCK这块正常取对应数据，其中HALO_SIZE块在row中是左右二边，如果是最左边和最右边需要考虑取不到的情况，我采用的逻辑对应opencv的边框填充REPLICATE模式，余下的块的HALO_SIZE块都不是对应当前线程组对应的图像块。column的上下块同理，可以看到最大核的大小限定在HALO_SIZEx2+WorkGroupSize.

其中在PC平台应用如上优化会有噪点，特别是核小的时候。

我分别针对filterRow/filterColumn做测试应用，发现只有filterColumn有问题，而代码我反复检测也没发现那有逻辑错误，更新逻辑查看filterColumn我在groupMemoryBarrier后，隔gl_WorkGroupSize.y的数据能拿到，但是行+1拿的是有噪点的，断定问题出在同步局部共享显存上，groupMemoryBarrier改为memoryBarrier还是不行，后改为barrier可行，按逻辑上来说，应该是用groupMemoryBarrier就行，不知是不是和硬件有关。

在1080P下取核半径为10的高斯模糊查看没有优化/优化的效果。

![avatar](../../images/gaussianA.png "gaussian image")

![avatar](../../images/gaussianB.png "gaussian image")

其中没优化的需要12.03ms,而优化后的是0.60+0.61=1.21ms,差不多10倍左右的差距.

### AdaptiveThreshold 自适应阈值化操作

[自适应阈值化操作](https://www.cnblogs.com/GaloisY/p/11037350.html)

效果图:

![avatar](../../images/adaptiveThreshold1.png "REPLICATE image")

### HarrisCornerDetection

[How to detect corners in a binary images with OpenGL?](https://dsp.stackexchange.com/questions/401/how-to-detect-corners-in-a-binary-images-with-opengl)

[Harris 角点检测](https://blog.csdn.net/u014485485/article/details/79056666)

genType step(float edge,genType x),step generates a step function by comparing x to edge.For element i of the return value, 0.0 is returned if edge >= x, and 1.0 is returned otherwise.

### 导向滤波(Guided Filter)

引导滤波是由何凯明等人于2010年发表在ECCV的文章《Guided Image Filtering》中提出的，后续于2013年发表了改进算法快速引导滤波的实现。它与双边滤波最大的相似之处，就是同样具有保持边缘特性。该模型认为，某函数上一点与其邻近部分的点成线性关系，一个复杂的函数就可以用很多局部的线性函数来表示，当需要求该函数上某一点的值时，只需计算所有包含该点的线性函数的值并做平均即可。这种模型，在表示非解析函数上，非常有用。

### Lookup

一张8x8x(格子(64x64)),把8x8垂直平放堆叠,可以理解成一个每面64像素的正方体,其中每个格子的水平方是红色从0-1.0,垂直方向是绿色从0-1.0,而对于正方体的垂直方向是蓝色从0-1.0.

颜色对应就很简单了,原图的蓝色确定是那二个格子(浮点数需要二个格子平均),红色找到对应格子的水平方向,绿色是垂直方向.

### AverageLuminanceThreshold

不同于GPUImage,先平均缩少3*3倍,然后读到CPU中计算平均亮度,然后再给下一层计算.

这步回读会浪费大量时间,我之前在GPU测试过,1080P的回读大约在2ms左右,就算少了9倍数据量,也需要0.2ms,再加上,回读CPU需要同步vulkan的cmd执行线程,早早的vkQueueSubmit,中断流程所导致的同步时间根据运行层的复杂度可能会比上面更长.

因此在这里,不考虑GPUImage的这种实现方式,全GPU流程处理,使用Reduce方式算图像的聚合数据(min/max/sum)等,然后保存结果到1x1的纹理中,现在实现效果在2070下1080P下需要0.08ms,比一般的普通计算层更短.

不过对于图像流(视频,拉流)来说,可以考虑在当前graph执行完vkQueueSubmit,然后VkReduceLayer层输出的1x1的结果,类似输出层,然后在需要用这结果的层,在下一桢之前把这个结果写入UNIFORM_BUFFER中,可能相比取纹理值更快,这样不会打断执行流程,也不需要同步,唯一问题是当前桢的参数是前面的桢的运行结果.

### 双边滤波bilateralFilter

[双边滤波bilateralFilter](https://zhuanlan.zhihu.com/p/127023952)

### CannyEdgeDetection

[Canny Edge Detection Canny边缘检测](https://blog.csdn.net/kathrynlala/article/details/82902254)

逻辑有点同HarrisCornerDetection,由多层构成.

其中第四层4.Hysteresis Thresholding正确实现方法逻辑应该类似:opencv_cudaimgproc canny.cpp/canny.cu里edgesHysteresis,不过逻辑现有些复杂,后面有时间修改成这种逻辑.

现暂时使用GPUImage里的简化逻辑.

### CGAColorspace CGA滤镜

[Color Graphics Adapter](https://en.wikipedia.org/wiki/Color_Graphics_Adapter)

[色彩调整之灰度、替换、深褐色、CGA滤镜](https://blog.csdn.net/h2282802627/article/details/114112435)

GPUImageCGAColorspaceFilter CGA滤镜。CGA全称是:Color Graphics Adapter，彩色图形适配器(关于CGA的更多资料请访问Color Graphics Adapter)。在320x200标准图形模式下，可以选择3种固定的调色板:

1.CGA 320×200 in 4 colors palette 0 (red, yellow, green, black background)
2.CGA 320×200 in 4 colors palette 1 (cyan, magenta, white, black background)
3.CGA 320×200 in 4 colors 3rd palette (tweaked), (cyan, red, white, black background)
