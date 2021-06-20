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
#include "AoceMetadata.h"
#include "AoceVideoDevice.h"
#include "AoceWindow.h"

namespace aoce {

typedef ITLayer<InputParamet> AInputLayer;
typedef ITLayer<OutputParamet> AOutputLayer;
// YUV 2 RGBA/RGBA 2 YUV 转换
typedef ITLayer<YUVParamet> IYUVLayer;
typedef ITLayer<MapChannelParamet> IMapChannelLayer;
typedef ITLayer<FlipParamet> IFlipLayer;
typedef ITLayer<TransposeParamet> ITransposeLayer;
typedef ITLayer<ReSizeParamet> IReSizeLayer;
typedef ITLayer<BlendParamet> IBlendLayer;

typedef ILTMetadata<bool> ILBoolMetadata;
typedef ILTMetadata<const char*> ILStringMetadata;

typedef ILTRangeMetadata<int32_t> ILIntMetadata;
typedef ILTRangeMetadata<float> ILFloatMetadata;

class IInputLayer : public AInputLayer {
   public:
    virtual ~IInputLayer(){};
    // inputCpuData(uint8_t* data)/inputGpuData()没有提供长宽,需要这个方法指定
    virtual void setImage(const ImageFormat& newFormat) = 0;
    virtual void setImage(const VideoFormat& newFormat) = 0;
    // 输入CPU数据,这个data需要与pipegraph同线程,因为从各方面考虑这个不会复制data里的数据.
    virtual void inputCpuData(uint8_t* data, bool bSeparateRun = false) = 0;
    virtual void inputCpuData(const VideoFrame& videoFrame,
                              bool bSeparateRun = false) = 0;
    virtual void inputCpuData(uint8_t* data, const ImageFormat& imageFormat,
                              bool bSeparateRun = false) = 0;
    virtual void inputGpuData(void* device, void* tex) = 0;
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

    virtual void outGLGpuTex(const GLOutGpuTex& outTex, uint32_t texType = 0,
                             int32_t outIndex = 0) = 0;
};

// 在AoceManager注册vulkan/dx11/cuda类型的LayerFactory
class LayerFactory {
   public:
    virtual ~LayerFactory(){};

   public:
    virtual IInputLayer* createInput() = 0;
    virtual IOutputLayer* createOutput() = 0;
    virtual IYUVLayer* createYUV2RGBA() = 0;
    virtual IYUVLayer* createRGBA2YUV() = 0;
    virtual IMapChannelLayer* createMapChannel() = 0;
    virtual IFlipLayer* createFlip() = 0;
    virtual ITransposeLayer* createTranspose() = 0;
    virtual IReSizeLayer* createSize() = 0;
    virtual IBlendLayer* createBlend() = 0;
};

class ILogObserver {
   public:
    virtual ~ILogObserver() = default;
    virtual void onLogEvent(int level, const char* message) = 0;
};

extern "C" {

ACOE_EXPORT void setLogObserver(ILogObserver* observer);

// 检查模块aoce_cuda/aoce_vulkan/aoce_win_mf/aoce_ffmpeg是否加载成功
ACOE_EXPORT bool checkLoadModel(const char* modelName);

// 得到分配GPU管线的工厂对象,请不要手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT PipeGraphFactory* getPipeGraphFactory(const GpuType& gpuType);

// 得到分配基本层的工厂对象,请不要手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT LayerFactory* getLayerFactory(const GpuType& gpuType);

// 得到对应设备的管理对象,请不要手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT IVideoManager* getVideoManager(const CameraType& cameraType);

// 得到分配媒体播放器的工厂对象,请不要手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT MediaFactory* getMediaFactory(const MediaType& mediaType);

// 得到直播模块提供的管理对象,请不要手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT ILiveRoom* getLiveRoom(const LiveType& liveType);

// 得到窗口管理对象,请不要手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT IWindowManager* getWindowManager(const WindowType& windowType);

// 得到窗口捕获对象,请不要手动释放这个对象,这个对象由aoce管理
ACOE_EXPORT ICaptureWindow* getWindowCapture(const CaptureType& captureType);

// 得到对应计算层的参数元数据信息(swig转换后的类,虽然继承关系还在,但是向下转换在运行时得不到相关信息)
ACOE_EXPORT ILMetadata* getLayerMetadata(const char* layerName);
ACOE_EXPORT ILGroupMetadata* getLGroupMetadata(ILMetadata* lmeta);
ACOE_EXPORT ILBoolMetadata* getLBoolMetadata(ILMetadata* lmeta);
ACOE_EXPORT ILStringMetadata* getLStringMetadata(ILMetadata* lmeta);
ACOE_EXPORT ILIntMetadata* getLIntMetadata(ILMetadata* lmeta);
ACOE_EXPORT ILFloatMetadata* getLFloatMetadata(ILMetadata* lmeta);
}

}  // namespace aoce