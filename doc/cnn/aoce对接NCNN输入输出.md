# aoce对接NCNN的Vulkan输入输出

接上篇[NCNN优化实时面部关键点检测](NCNN优化实时面部关键点检测.md)没有实现vulkan前期处理后的buffer直接和ncnn进行显存对接.

由[ncnn_Android_RobustVideoMatting](https://github.com/FeiGeChuanShu/ncnn_Android_RobustVideoMatting)这个项目来测试,其中ncnn接收512x512x4x3Byte的数据,输出512x512x4Byte的数据,经过直接使用VkBuffer对接,在PC平台2070下,平均13.9ms到12.7ms,可以说不明显,没有上篇改进效果好,并且现在中间临时变量通过GPU复制应该有问题,因此只用来记录下对应过程.

在上篇,从aoce的vkBuffer复制到ncnn的vkbuffer一直不成功,于是打开vulkan调试层,提示说操作ncnn的vkbuffer无效,很奇怪,我用ncnn里的VkCompute操作这个buffer正常,后面才忽然想到,aoce与ncnn二者都用的不是同一个VkDevice,搜索了下不同VkDevice之间复制VkBuffer后,暂时没找到有用信息,故从aoce_vulkan本身考虑,可以使用第三方库的vulkan环境替换aoce的vulkan环境.

在完成如上逻辑后,确实可以把aoce的vkBuffer复制到ncnn的vkbuffer了,但是结果并不正确,跟踪了ncnn相应输入逻辑发现,其中ncnn网络根据选项use_fp16_storage会应用fp16优化,输入CPU数据时会自动转化成对应的fp16GPU数据,而输入GPU数据时,需要自身来做处理,重新修改[NCNN优化实时面部关键点检测](NCNN优化实时面部关键点检测.md)的[shader](https://github.com/xxxzhou/aoce/blob/master/glsl/source/ncnnInMat.comp),支持fp16数据输出.

```glsl
#version 450

#if NCNN_FP16
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif

layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0) uniform sampler2D inSampler;
layout (binding = 1) buffer outBuffer{
#if NCNN_FP16
    float16_t dataOut[]; 
#else
    float dataOut[];
#endif
};

layout (std140, binding = 2) uniform UBO {    
    int outWidth;
    int outHeight;
    float meanX;
    float meanY;
    float meanZ;
    float meanW;
    float scaleX;
    float scaleY;
    float scaleZ;
    float scaleW;
#if NCNN_CROP
    float x1;
    float x2;
    float y1;
    float y2;
#endif
} ubo;

void main(){
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);    
    if(uv.x >= ubo.outWidth || uv.y >= ubo.outHeight){
        return;
    }      
#if NCNN_CROP
    vec2 isize = vec2(ubo.x2-ubo.x1,ubo.y2-ubo.y1);
    vec2 isuv = (vec2(uv)+vec2(0.5f))/vec2(ubo.outWidth,ubo.outHeight); 
    vec2 suv = isuv*isize+vec2(ubo.x1,ubo.y1); 
#else
    vec2 suv = (vec2(uv)+vec2(0.5f))/vec2(ubo.outWidth,ubo.outHeight); 
#endif 
    vec4 inColor = textureLod(inSampler,suv,0)*255.0f;  
    int size = ubo.outWidth*ubo.outHeight;
    int index = uv.y*ubo.outWidth+uv.x;
    vec4 mean = vec4(ubo.meanX,ubo.meanY,ubo.meanZ,ubo.meanW);
    vec4 scale = vec4(ubo.scaleX,ubo.scaleY,ubo.scaleZ,ubo.scaleW);
#if NCNN_FP16    
    f16vec4 result = f16vec4((inColor-mean)*scale);
#else
    vec4 result = (inColor-mean)*scale;
#endif
#if NCNN_BGR    
    dataOut[index] = result.b;
    dataOut[index+size] = result.g;
    dataOut[index+2*size] = result.r;
#endif
#if NCNN_RGB
    dataOut[index] = result.r;
    dataOut[index+size] = result.g;
    dataOut[index+2*size] = result.b;
#endif
}
```

这样就能直接输出fp16的数据,节省下CPU密集计算与显存/内存拷贝,再次输入VkMat,可以得到正确结果.

``` c++
bool VkNcnnInLayer::onFrame() {
    if (!bOneVkDevice) {
        memcpy(inVkMat.mapped_ptr(), outBuffer->getCpuData(),
               outBuffer->getBufferSize());
        // getNgParamet().vkAllocator->flush(inVkMat.data);
    }
    if (observer) {
        observer->onResult(inVkMat, inFormats[0]);
    }
    return true;
}
void FaceDetector::onResult(ncnn::VkMat& vkMat,
                            const ImageFormat& imageFormat) {
    if (!bInitNet) {
        return;
    }
    long long time1 = getNowTimeStamp();

    ncnn::Extractor netEx = net->create_extractor();
    ncnn::Mat boxMat, scoreMat, landmarkMat;

    netEx.input(0, vkMat);
    ...
}    
void FaceKeypointDetector::onResult(ncnn::VkMat& vkMat,
                                    const ImageFormat& imageFormat) {
    if (!bInitNet) {
        return;
    }
    long long time1 = getNowTimeStamp();
    // 得到当前面部区域
    FaceBox faceBox = {};
    ncnnInLayer->getInFaceBox(faceBox);

    ncnn::Extractor netEx = net->create_extractor();
    netEx.input("input_1", vkMat);
    ncnn::Mat out;
    netEx.extract("415", out);
    ...
} 
```

输出的结果,如上面的[ncnn_Android_RobustVideoMatting](https://github.com/FeiGeChuanShu/ncnn_Android_RobustVideoMatting),拿到的VkMat也是fp16的数据,如果拿的是Mat,则是fp32的数据,这个结果直接拿去渲染相关,则不走CPU那边.

```glsl
#version 450

#if NCNN_FP16
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#endif

layout (local_size_x = 16, local_size_y = 16) in;
layout (binding = 0) buffer inBuffer{
#if NCNN_FP16
    float16_t dataIn[]; 
#else
    float dataIn[];
#endif
};

layout (binding = 1, r8) uniform image2D outTex;

void main(){
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outTex);
    if(uv.x >= size.x || uv.y >= size.y){
        return;
    } 
    float vaule = float(dataIn[uv.y*size.x + uv.x]);
    imageStore(outTex,uv,vec4(vaule));  
}
```

针对其中清零与特殊复制(如目标buffer已有),根据ncnn::VkCompute仿写一个简单的VkCommand用来处理.

现如上逻辑有兴趣需要自己手动打开,主要因为如下二点.

1 ncnn的VkInstance没有放出来,并且VkInstance对应ppEnabledExtensionNames在win32下没有包含VK_KHR_WIN32_SURFACE_EXTENSION_NAME,所以不能创建vulkan渲染窗口,需要补起相关逻辑才可以在文件VkNcnnModule.cpp里设置NCNN_WIN32_VULKAN_INSTANCE为1,否则要么使用setVulkanContext传入physical_device/vkdeivce替换aoce_vulkan本身执行环境,这样ncnn/aoce可以使用GPU显存数据直接输入输出,输入输出的fp32/fp16转换全在GPU上进行,但是窗口只能用opencv的,要么不使用setVulkanContext替换aoce_vulkan的环境,ncnn/aoce只能通过CPU交换,可以使用vulkan渲染窗口.

2 android下的VkDevice的ppEnabledExtensionNames没包含VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,则不能直接使用VkImage与opengl的绑定纹理,导致现不能把结果给opengl直接渲染.

现使用ncnn直接替换aoce的Vulkan环境还有一些问题,主要是对应的扩展不同,但是如果不替换,aoce计算的vulkan buffer我暂时还没找到好方法可以和ncnn输入/输出对接.现阶段是拿到ncnn的源码,修改如上二点,可分别在window/android使用vulkan/opengl显示与全vulkan对接输入输出.

其中有些转换条件与逻辑还没跟清楚,欢迎大佬指出其中错误之处.
