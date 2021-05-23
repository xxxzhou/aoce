#pragma once
// 给外部用户使用

// 导出给外部用户使用,主要三种
// 1. C风格的结构,与C导出用来创建对应工厂/管理对象.
// 2. 纯净的抽像C++类,用户不要继承这些类,主要用来调用API.
// 3. 抽像C++类后缀Observer类的,用户继承针对接口处理回调.

#include "Aoce.h"
#include "AoceLayer.h"
#include "AoceLive.h"
#include "AoceMath.h"
#include "AoceMedia.h"
#include "AoceVideoDevice.h"

namespace aoce {

typedef ITLayer<InputParamet> AInputLayer;
typedef ITLayer<OutputParamet> AOutputLayer;
// YUV 2 RGBA 转换
typedef ITLayer<YUVParamet> IYUV2RGBALayer;
// RGBA 2 YUV 转换
typedef ITLayer<YUVParamet> IRGBA2YUVLayer;
typedef ITLayer<TexOperateParamet> ITexOperateLayer;
typedef ITLayer<TransposeParamet> ITransposeLayer;
typedef ITLayer<ReSizeParamet> IReSizeLayer;
typedef ITLayer<BlendParamet> IBlendLayer;

class IInputLayer : public AInputLayer {
   public:
    virtual ~IInputLayer(){};
    // inputCpuData(uint8_t* data)这个版本没有提供长宽,需要这个方法指定
    virtual void setImage(VideoFormat newFormat) = 0;
    // 输入CPU数据,这个data需要与pipegraph同线程,因为从各方面考虑这个不会复制data里的数据.
    virtual void inputCpuData(uint8_t* data, bool bSeparateRun = false) = 0;
    virtual void inputCpuData(const VideoFrame& videoFrame,
                              bool bSeparateRun = false) = 0;
    virtual void inputCpuData(uint8_t* data, const ImageFormat& imageFormat,
                              bool bSeparateRun = false) = 0;
};

class IOutputLayer : public AOutputLayer {
   public:
    virtual ~IOutputLayer(){};
    virtual void setObserver(IOutputLayerObserver* observer) = 0;
    // vk: contex表示vkcommandbuffer,texture表示vktexture
    // dx11: contex表示ID3D11Device,texture表示ID3D11Texture2D
    virtual void outVkGpuTex(const VkOutGpuTex& outTex,
                             int32_t outIndex = 0) = 0;

    virtual void outDx11GpuTex(void* device, void* tex) = 0;

#if __ANDROID__
    virtual void outGLGpuTex(const VkOutGpuTex& outTex, uint32_t texType = 0,
                             int32_t outIndex = 0) = 0;
#endif
};

// 在AoceManager注册vulkan/dx11/cuda类型的LayerFactory
class LayerFactory {
   public:
    virtual ~LayerFactory(){};

   public:
    virtual IInputLayer* crateInput() = 0;
    virtual IOutputLayer* createOutput() = 0;
    virtual IYUV2RGBALayer* createYUV2RGBA() = 0;
    virtual IRGBA2YUVLayer* createRGBA2YUV() = 0;
    virtual ITexOperateLayer* createTexOperate() = 0;
    virtual ITransposeLayer* createTranspose() = 0;
    virtual IReSizeLayer* createSize() = 0;
    virtual IBlendLayer* createBlend() = 0;
};

extern "C" {

ACOE_EXPORT void setLogObserver(ILogObserver* observer);

// 检查模块aoce_cuda/aoce_vulkan/aoce_win_mf/aoce_ffmpeg是否加载成功
ACOE_EXPORT bool checkLoadModel(const char* modelName);

// 得到分配GPU管线的工厂对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT PipeGraphFactory* getPipeGraphFactory(const GpuType& gpuType);

// 得到分配基本层的工厂对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT LayerFactory* getLayerFactory(const GpuType& gpuType);

// 得到对应设备的管理对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT IVideoManager* getVideoManager(const CameraType& cameraType);

// 得到分配媒体播放器的工厂对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT MediaFactory* getMediaFactory(const MediaType& mediaType);

// 得到直播模块提供的管理对象,请不会手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT ILiveRoom* getLiveRoom(const LiveType& liveType);
}

}  // namespace aoce