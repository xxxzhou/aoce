# GPUImage移植总结

项目github地址: [aoce](https://github.com/xxxzhou/aoce)

我是去年年底才知道有GPUImage这个项目,以前也一直没有在移动平台开发过,但是我在win平台有编写一个类似的项目[oeip](https://github.com/xxxzhou/oeip)(不要关注了,所有功能都移植或快移植到aoce里了),移动平台是大势所趋,开始是想着把oeip移植到android平台上,后面发现不现实,就直接重开项目,从头开始,从Vulkan到CMake,再到GPUImage,开发主力平台也从Visual Studio 2017换到VSCode了,这也算是前半年的总结了.

[Vulkan移植GPUImage(一)高斯模糊与自适应阈值](Vulkan移植GPUImage1.md)

[Vulkan移植GPUImage(二)Harris角点检测与导向滤波](Vulkan移植GPUImage2.md)

[Vulkan移植GPUImage(三)从A到C的滤镜](Vulkan移植GPUImage3.md)

[Vulkan移植GPUImage(四)从D到O的滤镜](Vulkan移植GPUImage4.md)

[Vulkan移植GPUImage(五)从P到Z的滤镜](Vulkan移植GPUImage5.md)

[CMake 常用命令](https://zhuanlan.zhihu.com/p/258118287)

[在Android用vulkan完成蓝绿幕扣像](https://zhuanlan.zhihu.com/p/348824878)

[android下vulkan与opengles纹理互通](https://zhuanlan.zhihu.com/p/302285687)

[Vulkan与DX11交互](https://zhuanlan.zhihu.com/p/349534525)

[PC的Vulkan运算层时间计算记录](PC平台Vulkan运算层时间记录.md)

[Vulkan移植GPUImage的Compute Shader总目录](https://github.com/xxxzhou/aoce/blob/master/glsl/source)

## 选择Vulkan的Compute Shader处理管线

当初选择Vulkan,一是越来越多设备与平台支持,且有独立的计算管线.

独立的计算管线在移植GPUImage里时好处如下.

1 避免很多UV生成类,如GPUImage里的GPUImageTwoInputFilter / GPUImageTwoInputCrossTextureSamplingFilter等等这种要么多个输入,要么需要查找周围点来生成不同UV,特别还有多个输入与需要周边UV结合,导致其中GPUImage中有很多类就是用来给FS提供UV.

2 不需要一个对应Vulkan渲染输出窗口,简单来说,你可以无窗口运行计算流程,并把结果直接对接win平台GUI32/DX11的CPU输出/GPU纹理,也可以在android中对接opengl es纹理,也可以方便对接引擎UE4/Unity3D.

3 计算管线可以利用局部共享显存,局部共享显存在那种需要查找周边多个点的情况能大幅提高性能,原则上来说,CS比渲染管线少PS之前的那一系列阶段,最新的硬件应该会比用VS+PS高吧?我用vulkan/cuda/dx11(原oeip实现)比较了下运行复杂计算管线的情况,cuda的GPU占比最低,vulkan其次,dx11会在cuda/vulkan的二倍以上.

不过缺点也有,其中有三个没移植GPUImage的功能,其中二个就是画多条线的,主要就是利用VS/PS渲染管线完成,暂时还没想出好的方法移植,还有一个图像2D-3D多角度转换利用VS/PS渲染管线也很方便,不过这个在独立的计算管线应该也好做.

## Vulkan数据处理流程

我定义主要实现要满足二点.

1. 计算流程可以多个输入/输出,每个节点可以多个输入输出,每个节点可以关闭打开,也可关闭打开此节点分支.

2. 别的用户能非常容易扩展自己的功能,就是自定义图像处理层的功能.

第一点,我受[FrameGraph|设计&基于DX12实现](https://zhuanlan.zhihu.com/p/147207161)启发,想到利用有向无环图来实现.在开始构建时,节点互相连接好.然后利用深度优先搜索自动排除到关闭自己/分支的节点,拿到最终的有效连接线,有向无环图可以根据有效连接线生成正确的执行顺序,然后检查每层节点与连接的节点的图像类型是否符合,检查成功后就初始化每层节点的资源,如果是Vulkan模块,所有层资源生成后,就会把所有执行命令填充到当前图层的VkCommandBuffer对象中,运行时执行VkCommandBuffer记录的指令.

在运行时,设定节点/分支是否可用,以及有些层参数改变会影响输出大小都会导致图层重启标记开启,用标记是考虑到更新参数层与执行GPU运算不在同一线程的情况,图层下次运行前,检测到重启标记开启,就会重新开始构建.

相关源码在[PipeGraph](../code/aoce/layer/PipeGraph.hpp)

而第二点,为了方便用户扩展自己的层,我需要尽可能的自动完善各种信息来让用户只专注需求实现.

对于运算层基类(BaseLayer)注意如下几个时序.

1. onInit/onInitNode 当层被加入PipeGraph前/后分别调用,在之后,会有个弱引用关联PipeGraph的PipeNode,同样,检查这个引用是否有效可以知道是否已经附加到PipeGraph上.

2. onInitLayer 当PipeGraph生成正确的有效连接线后,根据有效连接线重新连接上下层并生成正确执行顺序后,对各运算层调用.

3. onInitBuffer 每层输入检查对应连接层的输出的图像类型是否符合.

4. onFrame 每桢运行时调用.

5. onUpdateParamet 层的参数更新,时序独立于上面的4个点,设定要求随时可以调用.

相关源码在[BaseLayer](../code/aoce/layer/BaseLayer.hpp)

准确到Vulkan模块,Vulkan下的运算层基类(VkLayer)会针对BaseLayer提供更精确的Vulkan资源时序.

1. 初始化,一般指定使用的shader路径,UBO大小,更新UBO内数据.默认认为一个输入,一个输出,如果是多输入与多输出,可以在这指定.注意输入/输出个数一定要在附加在PipeGraph之前确定,相应数组会根据这二个值生成空间.

2. onInitGraph,当vklayer被添加到VkPipeGraph时上被调用.一般用来加载shader,根据输入与输出个数生成pipelineLayout,如果有自己逻辑,请override.默认指定输入输出的的图像格式为rgba8,如果不是,请在这指定对应图像格式.如果层内包含别的处理层逻辑,请在这添上别的处理层.

3. onInitNode,当onInitGraph后被添加到PipeGraph后调用.本身layer在onInitGraph后,onInitNode前添加到PipeGraph了,当层内包含别的层时,用来指定层内之间的数据如何链接.

4. onInitLayer,当PipeGraph根据连接线重新构建正确的执行顺序后.根据各层是否启用等,PipeGraph构建正确的各层执行顺序,在这里,每层都知道对应层数据的输入输出层,也知道输入输出层的大小.当前层的输入大小默认等于第0个输入层的输出大小,并指定线程组的分配大小,如果逻辑需要变化,请在这里修改.

5. onInitVkBuffer,当所有有效层执行完后onInitLayer后,各层开始调用onInitBuffer,其在自动查找到输入层的输出Texture,并生成本层的输出Texture给当前层的输出使用后调用.如果自己有Vulkan Buffer需要处理,请在onInitVkBuffer里处理.

6. onInitPipe,当本层执行完onInitVkBuffer后调用,在这里,根据输入与输出的Texture自动更新VkWriteDescriptorSet,并且生成ComputePipeline.如果有自己的逻辑,请override实现.

7. onCommand 当所有层执行完onInitBuffer后,填充vkCommandBuffer,vkCmdBindPipeline/vkCmdBindDescriptorSets/vkCmdDispatch 三件套.

8. onFrame 每桢处理时调用,一般来说,只有输入层或输出层override处理,用于把vulkan texture交给CPU/opengl es/dx11等等.

相关源码在[VkLayer](../code/aoce_vulkan/layer/VkLayer.hpp)

虽然列出有这么多,但是从我移植GPUImage里来看,很多层特别是混合模式那些处理,完全一个都不用重载,就只在初始化指定下glslPath就行了,还有许多层按上面设定只需要重载一到二个方法就不用管了.

其中Vulkan图层中,每个图层中包含一个VulkanContext对象,其有独立的VkCommandBuffer对象,这样可以保证每个图层在多个线程互不干扰,各个线程可以独立运行一个或是多个图层,对于cuda图层来说,每个图层也有个cudaStream_t对象,做到各个线程独立运行.

其中aoce_vulkan我定义了VkPipeGraph/VkLayer的实现,以及各个Vulkan对象的封装,还有输入/输出,包含RGBA<->YUV的转化这些基本的计算层,余下的GPUImage的所有层全在aoce_vulkan_extra完成,也算是对方便用户扩展自己的层的一个测试,说实话,在移植GPUImage到aoce_vulkan_extra模块过程中,我感觉以前存储的一些Vulkan知识已经快被我忘光了.

最后到这,用户实现自己的vulkan处理层,就不需要懂太多vulkan知识就能完成,只需要写好glsl源码,继承VkLayer,然后根据需求重载上面的一二个函数就行了,欢迎大家在这基础之上实现自己的运算层.

## 框架数据流程

数据提供现主要包含如下三种.

1. 摄像头,在win端,有aoce_win_mf模块提供,在android端,有aoce_android提供.

2. 对于多媒体文件(本地多媒体,RTMP等),由aoce_ffmpeg(win/android都支持)提供解码.

3. 直接非压缩的图像二进制数据.

数据处理模块现有aoce_cuda/aoce_vulkan模块处理,win端现支持这二个模块,而android端只支持aoce_vulkan模块.

如果数据提供的是桢数据,对应摄像头/多媒体模块都会解析到VideoFrame并给出回调,而在数据处理模块会有InputLayer层,专门用来接收上面三种数据.

而处理后数据会根据对应OutputLayer需要,导出CPU数据以及GPU数据对接对应系统常用渲染引擎对应纹理上,如在win端,aoce_cuda/aoce_vulkan模块的OutputLayer都支持直接导致到对应DX11纹理,而在android上,aoce_vulkan能直接导致到对应opengl es纹理上,这样就能直接与对应引擎(UE4/Unity3D)底层进行对接.

## 导出给用户调用

在重新整理了框架与结构,完善了一些内容,API应该不会有大的变动了,现开始考虑外部用户使用.

在框架各模块内部,引用导出的类不要求什么不能用STL,毕竟肯定你编译这些模块肯定是相同编译环境,但是如果导出给别的用户使用,需要限制导出的数据与格式,以保证别的用户与你不同的编译环境也不会有问题.

配合CMake,使用install只导出特殊编写的.h头文件给外部程序使用,这些头文件主要包含如下三种类型.

1. C风格的结构,C风格导出帮助函数,与C风格导出用来创建对应工厂/管理对象.

2. 纯净的抽像类,不包含任何STL对象结构,主要用来调用API,用户不要继承这些类.

3. 后缀为Observer的抽像类,用户继承针对接口处理回调.
