# Vulkan移植GPUImage(五)从P到Z的滤镜

现[aoce_vulkan_extra](https://github.com/xxxzhou/aoce)把GPUImage里从P到Z的大部分滤镜用vulkan的ComputeShader实现了,也就是最后一部分的移植,整个过程相对前面来说比较简单,大部分我都是直接复制以前的实现改改就行了,还是列一些说明.

## PerlinNoise 柏林燥声生成一张静态图

柏林燥声的原理网上很多讲解的,用于生成平滑的图案,其实可以稍微改下,如加个如时间戳,就可变成一张平滑的动态图.

## PinchDistortion 收缩,凹面镜

实现类似BulgeDistortion(鱼眼效果),都是针对UV变形,BulgeDistortion在设定圆心处UV缩小,意思在原圆心处相同的地方取更近的像素,这样就导致图像看起来向外扩张.

而PinchDistortion在设定圆心处,周边UV放大,意思在圆心处相同的地方取更远的UV像素,就导致图像看起来向里缩.

## PoissonBlend 泊松融合

[图像融合之泊松编辑(Poisson Editing)(2)](https://blog.csdn.net/u011534057/article/details/68922319)

GPUImage的实现应该是一种简化实现,后面会移植按上面原理求泊松重建方程的实现.

现暂时按照GPUImage里来实现,他的实现比较简单,唯一麻烦的,需要第一张输入图与输出图Ping-pong来回处理,不同于前面savefamelayer的实现,他需要在一桢中来回循环读写,刚开始想的是如何把当前索引当做UBO传入shader,但是在一桢中把一个UBO更新多次,首先不知道能否这样实现,就算能,这种实现并不好,在这引入Vulkan里的PushConstant的概念,能完美解决这个问题,首先vkCmdPushConstants也是插入到CommandBuffer中,在提交给GPU的时候,也是确定的数值,这样每桢多次循环就可以用PushConstant来表明对应次数.感觉在渲染一批次多模型时用来指定索引也不错啊.

```C++
void VkPoissonBlendLayer::onCommand() {
    // ping-pong 单数次
    paramet.iterationNum = paramet.iterationNum / 2 * 2 + 1;
    for (int32_t i = 0; i < paramet.iterationNum; i++) {
        int32_t pong = i % 2;
        inTexs[0]->addBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_READ_BIT);
        inTexs[1]->addBarrier(
            cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            pong == 0 ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_WRITE_BIT);
        outTexs[0]->addBarrier(
            cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            pong == 0 ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT);
        vkCmdPushConstants(cmd, layout->pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t),
                           &pong);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computerPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                layout->pipelineLayout, 0, 1,
                                layout->descSets[0].data(), 0, 0);
        vkCmdDispatch(cmd, sizeX, sizeY, 1);
    }
}
```

```glsl
#version 450

layout (local_size_x = 16, local_size_y = 16) in;// gl_WorkGroupSize
layout (binding = 0, rgba8) uniform image2D inTex;
layout (binding = 1, rgba8) uniform readonly image2D inTex1;
layout (binding = 2, rgba8) uniform image2D outTex;

layout (std140, binding = 3) uniform UBO {
    float percent;
} ubo;

layout(push_constant) uniform pushBlock {
    int pong;
} constBlock;

const ivec2 centerT[4] = {
    ivec2(1,0),
    ivec2(-1,0),
    ivec2(0,1),
    ivec2(0,-1)
};

void main(){
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(inTex);
    if(uv.x >= size.x || uv.y >= size.y){
        return;
    }     
    vec4 center = vec4(0);
    if(constBlock.pong == 0){
        center = imageLoad(inTex,uv);
    }else{
        center = imageLoad(outTex,uv);
    }
    vec4 center1 = imageLoad(inTex1,uv);  
    vec4 sum = vec4(0);
    vec4 sum1 = vec4(0);
    for(int i = 0; i < 4; i++){
        ivec2 cuv = uv + centerT[i];
        cuv = max(ivec2(0),min(cuv,size));
        if(constBlock.pong == 0){
            sum += imageLoad(inTex,cuv);
        }else{
            sum += imageLoad(outTex,cuv);
        }
        sum1 += imageLoad(inTex1,cuv);
    }
    vec4 mean = sum / 4.0;
    vec4 diff1 = center1 - sum1 /4.0;
    vec4 grad = mean + diff1;
    
    vec4 result = vec4(mix(center.rgb,grad.rgb,center1.a * ubo.percent),center.a);    
    if(constBlock.pong == 0){
        imageStore(outTex,uv,result);  
    }else{
        imageStore(inTex,uv,result);  
    }
}
```

在N卡2070上,一次迭代在0.2-0.3ms之间.

这个后面在内部又加了一个节点,用来保存第一个输入,应该这个层会改变第一个输入的内容,而别的层可能还需要这个输入,所以添加一层保存第一层输入.

## PrewittEdgeDetection

和SobelEdgeDetection没多大区别,算子不同,其实算子区别也不大,垂直与水平的正对方向上一个是1,一个是2,别的都没啥区别.

## RGBDilation/RGBErosion

把原来的Dilation/Erosion的glsl文件添加下编译符,针对处理下,取第一个横向的代码贴出来.

```glsl
#version 450

layout (local_size_x = 16, local_size_y = 16) in;// gl_WorkGroupSize

#if CHANNEL_RGBA
layout (binding = 0, rgba8) uniform readonly image2D inTex;
layout (binding = 1, rgba8) uniform image2D outTex;
#elif CHANNEL_R8
layout (binding = 0, r8) uniform readonly image2D inTex;
layout (binding = 1, r8) uniform image2D outTex;
#endif

layout (binding = 2) uniform UBO {
	int ksize;	    
} ubo;

#if EROSION 
    #define OPERATE min
    #if CHANNEL_RGBA
        #define INIT_VUL vec4(1.0)
    #elif CHANNEL_R8
        #define INIT_VUL 1.0f
    #endif
#endif

#if DILATION 
    #define OPERATE max
    #if CHANNEL_RGBA
        #define INIT_VUL vec4(0.0)
    #elif CHANNEL_R8
        #define INIT_VUL 0.0f
    #endif
#endif

#if IS_SHARED
// 限定最大核为32

#if CHANNEL_RGBA
    shared vec4 row_shared[16][16*3];
#elif CHANNEL_R8
    shared float row_shared[16][16*3];
#endif

void main(){
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(inTex);
    if(uv.x >= size.x || uv.y >= size.y){
        return;
    } 
    ivec2 locId = ivec2(gl_LocalInvocationID.xy);
    for(int i = 0; i < 3; i++){
        uint gIdx = max(0,min(uv.x+(i-1)*16,size.x-1));
        #if CHANNEL_RGBA
            row_shared[locId.y][locId.x + i*16] = imageLoad(inTex,ivec2(gIdx,uv.y));
        #elif CHANNEL_R8
            row_shared[locId.y][locId.x + i*16] = imageLoad(inTex,ivec2(gIdx,uv.y)).r;
        #endif   
    }
    memoryBarrierShared();
	barrier();
    #if CHANNEL_RGBA
        vec4 result = INIT_VUL;
    #elif CHANNEL_R8
        float result = INIT_VUL;
    #endif
    for(int i =0; i < ubo.ksize; i++){
        int ix = locId.x - ubo.ksize/2 + i;
        #if CHANNEL_RGBA
            vec4 fr = row_shared[locId.y][16 + ix];
            result = OPERATE(fr,result);
        #elif CHANNEL_R8
            float fr = row_shared[locId.y][16 + ix];
            result = OPERATE(fr,result);
        #endif
    }
    imageStore(outTex, uv, vec4(result)); 
}

#else

void main(){
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(inTex);
    if(uv.x >= size.x || uv.y >= size.y){
        return;
    } 
    #if CHANNEL_RGBA
        vec4 result = INIT_VUL;
    #elif CHANNEL_R8
        float result = INIT_VUL;
    #endif
    for(int i = 0; i< ubo.ksize; ++i){
        int x = uv.x-ubo.ksize/2+i;
        x = max(0,min(x,size.x-1));
        #if CHANNEL_RGBA
            vec4 r = imageLoad(inTex,ivec2(x,uv.y));
            result = OPERATE(result,r);
        #elif CHANNEL_R8
            float r = imageLoad(inTex,ivec2(x,uv.y)).r;
            result = OPERATE(result,r);
        #endif
    }
    imageStore(outTex, uv, vec4(result));     
}

#endif
```

下面对应的Closing/Opening都不用改,加个参数表示是R8/RGBA8就行.

## ShiTomasiFeatureDetection

和Noble角点检测一样,都和修改HarrisCornerDetection中的角点计算方式,别的流程一样.

## SingleComponentGaussianBlur 单通道高斯模糊

在最开始设计GaussianBlur时,就支持R8/RGBA8/RGBA32F这几种,要添加也只需要修改glsl的编译符,添加对应逻辑就行.

## SobelEdgeDetection/Sketch Sobel边缘检波/草图素描效果

SobelEdgeDetection利用Sobel算子计算边缘,而Sketch就是SobelEdgeDetection结果的反转.

下面的ThresholdEdgeDetection/ThresholdSketch在这二个基础上加了个Threshold用来确定结果是0还是1.

## SmoothToon 高斯模糊后的卡通效果

Toon是卡通效果,而SmoothToon就是先对输入图像高斯模糊后,然后再应用卡通效果.

## VoronoiConsumer 接收Voronoi映射,并使用该映射过滤传入的图像

二个输入,第二个输出需要长宽同为2^n的相同整数,根据第一张图中的UV,对应第二张图中的值,生成一个UV,然后取第一张图值,有点类似lookup映射.

## ZoomBlur 将定向运动模糊应用于图像

实现大致同MotionBlur,取周边点的算法稍微不同.

## 中间没有移植的滤镜统计

1. GPUImageParallelCoordinateLineTransform,同上篇文章里的LineGenerator,还没找到合适的使用GPGPU高效画线方法.

2. GPUImageSolidColorGenerator,在vulkan里,完全可以用vkCmdClearColorImage替换,效率更高,封装实现方法就在基类VkLayer::clearColor里.

3. GPUImageToneCurveFilter暂时不移植.看实现有点类似Lookup,但是需要根据传入的数据生成一张查找表.

4. GPUImageTransformFilter GPUImage利用顶点着色器,可以进行更多视角转换,现aoce_vulkan模块里有上下,左右,以及90/270转换,这个后面再仔细考虑下如何完善.

## 归类

让[GPUImage](https://gitee.com/xudoubi/GPUImage)里根据里分四类,在API导出头文件VkExtraExport.hpp里根据这四类重新排下,加下注释,如果更新参数是结构,相应结构添加下对应每个参数注释.

在其layer文件夹,对多个类进行合并,如几乎所有混合模式的类,实现都在glsl上,几乎不需要针对基类VkLayer做任何修改,所以都合并在VkBlendingModeLayer文件里,色彩调整/视觉效果也合并一些普通的类在对应的VkColorAdjustmentLayer/VkVisualEffectLayer中,而图像处理的类相对来说会复杂点,大部分都是分散到各个集合中,如其中的Dilation/Erosion/Closing/Opening合并到VkMorphLayer文件中,边缘检测的几个类实现在VkEdgeDetectionLayer中.
