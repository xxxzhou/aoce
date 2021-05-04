# Vulkan移植GpuImage(三)从D到O的滤镜

现把D到O的大部分滤镜用vulkan的ComputeShader实现了,列举其中一些有点特殊的说明.

## GaussianBlurPosition 指定区域高斯模糊

没有按照GPUImage里的方式实现,按照类似GaussianSelectiveBlur方式实现,一张高斯模糊图,一张原图,二图进行混合,这种实现方式更灵活(模糊半径等参数),并且并不会降低性能.

## SphereRefraction 环境映射

[环境映射技术漫谈](https://zhuanlan.zhihu.com/p/144438588)

和GlassSphere一样,主要就是球形坐标系与UV坐标转化,刚开始完成后,我测试效果,发现圈外在闪烁,后面发现没把圈外置0,withinSphere表示圈外圈内.

## Halftone 半色调效果,类似新闻打印

这个类继承PixellateFilter(马赛克),顺便也把这个实现了,他们参数一样,实现主要是把一小快矩形范围的UI统一成一个,然后处理.

## HighPass 高通滤波,检测图像像素变动

实现这个类,首先要实现LowPass,这个滤境有点特殊,需要保存上一桢的传入图像,然后比较当前桢与上一桢.

我设定运行图使用的有向无循环图,中间可以自动排除不可用节点并自动链接下层,所以不能有回环线,从一层保存一桢然后使用再拿回来使用,这样构成回环,所以我定义VkSaveFrameLayer层为bInput = true,告诉外面不需要自动连接别层输入,在相应接口手动指定别的层需要保存的纹理.

最开始我直接调用vkCmdCopyImage,发现在Nsight显示占用0.3ms-0.7ms左右的时间,从逻辑上来说,应该不可能啊,这个copy应该比最简单的运算层占去的时间要更少才对,于是我测试了二种方案,对应参数bUserPipe,表示用不用管线,用管线控制到0.2ms内,用vkCmdCopyImage在0.3ms以上,以后找下资料看看是什么问题.

LowPass有二个输入,然后就是第一输入节点是上层,第二个输入节点是SaveFrameLayer,在LowPass运行前,SaveFrameLayer提供输入,在LowPass后(混合前后桢,可以去掉类似摩尔纹的东东),把结果又保存在SaveFrameLayer里.

HighPass就是比较与LowPass的差异,可用来显示突示变化的像素.

相应源码,有兴趣可以自己查看[VkLowPassLayer](../code/aoce_vulkan_extra/layer/VkLowPassLayer.cpp)

## Histogram 直方图

GPUImage通过回读到CPU然后计算,这种方式直接放弃,查看相应opencv cuda模块类似实现.

线程组定成256个,然后使用数值对应索引位置+1,结果就是索引上的数值表示对应值的个数.

有二种方式,保存到临时buffer,二是用原子集合(int/uint),这里因为int满足,使用原子集合.

保存到临时buffer,我在开始四通道时使用类似reduce2分二段,不用原子操作的方式,但是效果并不好,一是第一次把16*16块的方式转换成对应的一个个直方图,模块并没的缩小,导致第二块把这一个直方图通过for加在一起需要循环1920/16x1080/16(假设是1080P的图),这个会花费超过2ms,这种方式就pass掉,我直接使用原子操作导出四个图然后再结合都比这个快.

一通道在0.18ms左右,四通道在0.52ms+0.01ms,比0.18*4快些.

直方图的显示可以使用VkAlphaShow2Layer显示,没有用GPUImage里的显示方式,其对应输出分别为256x1x(R32UI/RGBA32UI)纹理,自己用glsl显示各种需求应该更方便.要输出到CPU的话,直接接一个VkOutputLayer也方便.

## iOSBlur 一种特定模糊实现

组合downSampler/saturation/gaussian blur/luminance range/upSampler几个层的输出效果就是.

## Kuwahara 实现类似油画风格效果

在GPUImage上最开始就说了,只适合静止图像,因为很耗性能.原理比较简单,查找当前像素周边四个方向区域最小的区域.

因其算法特点,卷积分离没有使用,只使用了局部共享显存优化,在PC 2070N卡1080P下,半径5需要3.3ms,半径10需要9.7ms.

测试在手机Redmi 10X Pro 用半径3在720P下非常流畅,可以满足30fps运行.

下面专门优化的Kuwahara3x3就没用了,不可能比得了使用局部共享显存优化方案.

## Laplacian 使用拉普拉斯算子查找边缘

比较常见,一个像素与一个特定3x3矩阵的结果,单独拿出来说,是因为在ubo中,直接使用mat3/mat4,在CPU中使用类似的的Mat3x3,Mat4x4对齐会有问题,并且我在Nsight查看到对应的UBO里放着正确的mat3对应数据,但是结果就是不对.最不容易出错的方法就是在ubo中全使用单个float+std140表示类似vec3/vec4/mat3/mat4,在CPU端不需要特殊处理结构,和ubo一样顺序就行,这样最不容易出现CPU-GPU中的UBO对齐问题.

## Lookup 颜色查找表

这个在前面[Vulkan移植GpuImage(三)从A到C的滤镜](https://zhuanlan.zhihu.com/p/364888786)已经说过,但是感觉由用户自己来接一个输入层不太方便,因为修改逻辑,内置一个输入层,提供loadLookUp方法,由用户提供三通道/四通道512x512x(3/4)的lookup图像数据就行.

下面的MissEtikate导入lookup_miss_etikate图像数据就行,本框架现没有读各种图像格式的模块与第三方库.

## Median 中值滤波

[中值滤波原理及其C++实现与CUDA优化](https://zhuanlan.zhihu.com/p/355266029)

开始和Kuwahara一样,使用局部共享显存来优化,在结合上面文章里的直方图方式,但是结果并不好,详细情况可以看[PC平台Vulkan运算层时间记录](https://github.com/xxxzhou/aoce/tree/master/doc/PC平台Vulkan运算层时间记录.md)记录.

总的来说,优化了个寂寞,还是移植了GPUImage里的3x3的实现,虽然核大会导致排序也是指数增长,但是这次优化明显不成功,只能说是暂时可用大于3核的情况,后续找找更多的资料试试改进.

看了下相关opencv cuda里的median里的写法,在前面会声明一个图像元素*256的buffer?嗯,这个buffer是为了替换里面单个线程uint hist[256]?在CS里,一个线程声明大堆栈数据会导致性能问题吗?先暂停,opencv cuda这个算法写的太麻烦了,后面有时间研究.

## MotionBlur 运动模糊

在没看GPUImage实现前,我还在想HighPass很像运动模糊,但是其在GPUImage里,就是简单的给定一个方向,然后由这个方向模糊,并没有前后桢的叠加比较,因其实现简单,所以也移植了.

## MotionDetector 运动检测

这个倒是和HighPass很像,区别在于最后会聚合统计给CPU输出,这个确实有必要,所以在本MotionDetector实现中,集成一个VkOutputLayer,并转化输出数据与GPUImage类似,可以看到运动的大小统计.

``` c++
class VkMotionDetectorLayer : public VkLayer, public MotionDetectorLayer {
    AOCE_LAYER_QUERYINTERFACE(VkMotionDetectorLayer)
   private:
    std::unique_ptr<VkLowPassLayer> lowLayer = nullptr;
    std::unique_ptr<VkReduceLayer> avageLayer = nullptr;
    std::unique_ptr<VkOutputLayer> outLayer = nullptr;

   public:
    VkMotionDetectorLayer();
    virtual ~VkMotionDetectorLayer();

   private:
    void onImageProcessHandle(uint8_t* data, ImageFormat imageFormat,
                              int32_t outIndex);

   protected:
    virtual void onUpdateParamet() override;
    virtual void onInitGraph() override;
    virtual void onInitNode() override;
};

VkMotionDetectorLayer::VkMotionDetectorLayer(/* args */) {
    glslPath = "glsl/motionDetector.comp.spv";
    inCount = 2;

    lowLayer = std::make_unique<VkLowPassLayer>();
    avageLayer = std::make_unique<VkReduceLayer>(ReduceOperate::sum);
    outLayer = std::make_unique<VkOutputLayer>();

    paramet = 0.5f;
    lowLayer->updateParamet(paramet);
    outLayer->setImageProcessHandle(std::bind(
        &VkMotionDetectorLayer::onImageProcessHandle, this, _1, _2, _3));
}

void VkMotionDetectorLayer::onImageProcessHandle(uint8_t* data,
                                                 ImageFormat imageFormat,
                                                 int32_t outIndex) {
    if (onMotionEvent) {
        vec4 motion = {};
        memcpy(&motion, data, sizeof(vec2));
        onMotionEvent(motion);
    }
}

VkMotionDetectorLayer::~VkMotionDetectorLayer() {}

void VkMotionDetectorLayer::onUpdateParamet() {
    lowLayer->updateParamet(paramet);
}

void VkMotionDetectorLayer::onInitGraph() {
    VkLayer::onInitGraph();
    pipeGraph->addNode(lowLayer->getLayer());
    pipeGraph->addNode(avageLayer.get())->addNode(outLayer->getLayer());
}

void VkMotionDetectorLayer::onInitNode() {
    lowLayer->addLine(this, 0, 1);
    this->addLine(avageLayer.get());
    setStartNode(this, 0);
    setStartNode(lowLayer.get());
}
```

顺便还通过CPU的输出,查到了一个reduce2.comp里divup的低级错误.

## NobleCornerDetection Noble角点检测

这个GPUImage中,实现方式和HarrisCornerDetection几乎一样,就是角点的选择计算方式有点不同,会导致比HarrisCornerDetection多检测很多角点出来.

## Opening 开运算,先侵蚀后膨胀

由侵蚀与膨胀,专门提供一个类GroupLayer,本身不提供任何计算,只组合别的运算层.

## 中间没有移植的滤镜统计

1. FASTCornerDetection,其GPUImage里并没有实现,只是写个声明.

2. JFAVoronoi,看到其中需要CPU定循环多次GPU运行,其中UV与颜色对应相关代码暂时不理解,先放着,后期找更多资料确定移植方法.

3. LanczosResampling,GPUImage的实现好像和其原理差别有点大,应该是特化实现,暂不移植.

4. LineGenerator这个现在还没想到如何能在GPGPU中高效画多条线,普通的渲染管线方式倒是方便实现,后面查找资料确定移植方法.

5. 直方图的显示,其在VkHistogramLayer里会输出分别为256x1x(R32UI/RGBA32UI)纹理,自己根据你的需求去写相应glsl显示更合适.

其中从D到O的因为前面实现一些层的时候,有一些已经实现过,也不在本文里说明,现在GPUImager移植进度大约有60%,相应的效果可以在[vulkanextratest](../samples/vulkanextratest),win端修改Win32.cpp,android修改Android.cpp查看对应平台效果,等所有效果移植完成后会写配套专门的UI界面查看.
