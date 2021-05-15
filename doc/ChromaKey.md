# 在Android用vulkan完成蓝绿幕扣像

## 效果图(1080P处理)

[![在Android用vulkan完成蓝绿幕扣像](https://img2020.cnblogs.com/blog/81011/202102/81011-20210207101759253-1217562430.png)](https://video.zhihu.com/video/1339983217793961984)

因为摄像头开启自动曝光,画面变动时,亮度变化导致扣像在转动时如上.

源码地址[vulkan_extratest](https://github.com/xxxzhou/aoce/blob/master/android/vulkanextratest)

这个demo主要测试二点,一是测试ndk camera集成效果,二是本项目对接外部实现的vulkan层是否方便,用于以后移植GPUImage里的实现.

我简化了在[android下vulkan与opengles纹理互通](https://zhuanlan.zhihu.com/p/302285687)里的处理,没有vulkan窗口与交换链这些逻辑,只用到vulkan compute shader计算管线得到结果然后交换给opengl里的纹理.

## NDK Camera集成

主要参考 [NdkCamera Sample](https://github.com/android/ndk-samples/tree/master/camera)的实现,然后封装成满足Aoce定义设备接口.

说下遇到的坑.

1. AIMAGE_FORMAT_YUV_420_888 可能是YUV420P,也可能是NV12,需要在AImageReader_ImageListener里拿到image通过AImage_getPlanePixelStride里的UV的plan是否为1来判断是否为YUV420P,或者看data[u]-data[y]=1来看是否为NV12.具体可以看getVideoFrame的实现.

2. AImageReader_new里的maxImages比较重要,简单理解为预先申请几张图,这个值越大,显示越平滑.AImageReader_new如果不开线程,则图像处理加到这个线程里,导致读取图像变慢.打开线程处理,我用的Redmi K10 pro,可以读4000x3000,在AImageReader_ImageListener回调不做特殊处理,如下错误,首先是Unable to acquire a lockedBuffer, very likely client tries to lock more than. 可以看到,运行四次后报的,就是我设的maxImages,通过比对代码逻辑,应该是AImageReader_new读四次后,我还没处理完一桢,没有AImage_delete,也就读不了数据了. 然后检查 AImageReader_acquireNextImage 这个状态,不对不读,然后继续引发读取不可用内存问题,分析应该是处理数据的乱序线程AImage_delete可能释放别的处理线程上的image,然后处理图像线程上加上lock_guard(mutex),不会引发问题,但是会导致每maxImages卡一下,可以理解,读的线程快,处理的慢,后面想了下,直接让thread.join,图片读取很大时慢(比不开线程要快很多,4000*3000快二倍多,平均45ms),但是平滑的,暂时先这样,后面看能不能直接拿AImage的harderbuffer去处理,让处理速度追上读取速度.

## Chroma Key

如上所说,项目对接外部实现的vulkan层是否方便,在这重新生成一个模块aoce_vulkan_extra,在这我选择[UE4 Matting](https://www.unrealengine.com/en-US/tech-blog/setting-up-a-chroma-key-material-in-ue4)里的逻辑来测试,因为这个逻辑非常简单,也算让我对手机的性能有个初步的了解.

首先把相关逻辑整理下,UE4上有相关节点,看下实现整理成glsl compute shader实现.

```glsl
#version 450

// https://www.unrealengine.com/en-US/tech-blog/setting-up-a-chroma-key-material-in-ue4

layout (local_size_x = 16, local_size_y = 16) in;// gl_WorkGroupSize
layout (binding = 0, rgba8) uniform readonly image2D inTex;
layout (binding = 1, rgba8) uniform image2D outTex;

layout (std140, binding = 2) uniform UBO {
    // 0.2 控制亮度的强度系数
    float lumaMask;
    float chromaColorX;
    float chromaColorY;
    float chromaColorZ;
    // 用环境光补受蓝绿幕影响的像素(简单理解扣像结果要放入的环境光的颜色)
    float ambientScale;
    float ambientColorX;  
    float ambientColorY; 
    float ambientColorZ;   
    // 0.4
    float alphaCutoffMin;
    // 0.5
    float alphaCutoffMax;
    float alphaExponent;
    // 0.8
    float despillCuttofMax;
    float despillExponent;
} ubo;

const float PI = 3.1415926;

vec3 extractColor(vec3 color,float lumaMask){   
    float luma = dot(color,vec3(1.0f));
    // 亮度指数
    float colorMask = exp(-luma*2*PI/lumaMask);
    // color*(1-colorMask)+color*luma
    color = mix(color,vec3(luma),colorMask);
    // 生成基于亮度的饱和度图    
    return color / dot(color,vec3(2.0));
}

void main(){
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outTex);    
    if(uv.x >= size.x || uv.y >= size.y){
        return;
    }    
    vec3 inputColor = imageLoad(inTex,uv).rgb;   
    vec3 chromaColor = vec3(ubo.chromaColorX,ubo.chromaColorY,ubo.chromaColorZ);
    vec3 ambientColor = vec3(ubo.ambientColorX,ubo.ambientColorY,ubo.ambientColorZ);
    vec3 color1 = extractColor(chromaColor,ubo.lumaMask);
    vec3 color2 = extractColor(inputColor,ubo.lumaMask);
    vec3 subColor = color1 - color2;
    float diffSize = length(subColor);
    float minClamp = diffSize-ubo.alphaCutoffMin;
    float dist = ubo.alphaCutoffMax - ubo.alphaCutoffMin;
    // 扣像alpha
    float alpha= clamp(pow(max(minClamp/dist,0),ubo.alphaExponent),0.0,1.0);
    // 受扣像背景影响的颜色alpha
    float inputClamp = ubo.despillCuttofMax - ubo.alphaCutoffMin;
    float despillAlpha = 1.0f- clamp(pow(max(minClamp/inputClamp,0),ubo.despillExponent),0.0,1.0);
    // 亮度系数
    vec3 lumaFactor = vec3(0.3f,0.59f,0.11f);    
    // 添加环境光收益
    vec3 dcolor = inputColor*lumaFactor*ambientColor*ubo.ambientScale*despillAlpha;
    // 去除扣像背景
    dcolor -= inputColor*chromaColor*despillAlpha;
    dcolor += inputColor;    
    // 为了显示查看效果,后面屏蔽
    dcolor = inputColor*alpha + ambientColor*(1.0-alpha);
    imageStore(outTex,uv,vec4(dcolor,alpha)); 
}
```

这里面代码最后倒数第二句实现混合背景时去掉,在这只是为了显示查看效果.

然后引用aoce_vulkan里给的基类VkLayer,根据接口完成本身具体实现,相关VkChromKeyLayer的实现可以说是非常简单,至少我认为达到我想要的方便.

还是一样,先说遇到的坑,

1. 开始在glsl中的UBO,我特意把一个float,vec3放一起,想当然的认为是按照vec4排列,这里注意,vec3不管前后接什么,大部分结构定义下,都至少占vec4,所以后面为了和C++结构align一样,全部用float.

2. 层启用/不启用会导致整个运算graph重置,一般情况下,运算线程与结果输出线程不在一起,在重置时,运算线程相关资源会重新生成,而此时输出线程还在使用相关资源就会导致device lost错误,在这使用VkEvent用来表示是否在资源重置中.

然后就是与android UI层对接,android的UI没怎么用过,丑也就先这样吧.
