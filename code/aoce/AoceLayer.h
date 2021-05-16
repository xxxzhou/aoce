#pragma once
#include "Aoce.h"

// 导出给外部用户使用,纯净的抽像C++类.
// 请用户不要继承这里的类实现逻辑,找到对应基类实现继承

namespace aoce {

class IBaseLayer;
class IPipeGraph;
class ILayer;

struct InputParamet {
    int32_t bCpu = true;
    int32_t bGpu = false;
};

struct OutputParamet {
    int32_t bCpu = true;
    int32_t bGpu = false;
};

struct VkOutGpuTex {
    // vkcommandbuffer
    void* commandbuffer = nullptr;
#if WIN32
    void* image = nullptr;
#elif __ANDROID__
    uint64_t image = 0;
#endif
    int width = 1920;
    int height = 1080;
};

class IBaseLayer {
   public:
    virtual ~IBaseLayer(){};
    virtual const char* getMark() = 0;
    virtual bool bAttachGraph() = 0;
    virtual void setVisable(bool bvisable) = 0;
    virtual void setEnable(bool benable) = 0;
    virtual int32_t getGraphIndex() = 0;
    // 如果层有多个输入,可能不同输入对应不同层内不同层
    // index表示输入节点索引,node表示层内层节点,toInIndex表示对应层内层输入位置
    virtual void setStartNode(IBaseLayer* node, int32_t index = 0,
                              int32_t toInIndex = 0) = 0;
    virtual void setEndNode(IBaseLayer* node) = 0;
    virtual IBaseLayer* addNode(IBaseLayer* layer) = 0;
    virtual IBaseLayer* addNode(ILayer* layer) = 0;
    virtual IBaseLayer* addLine(IBaseLayer* to, int32_t formOut = 0,
                                int32_t toIn = 0) = 0;

   protected:
    // ITLayer类的所有实例化, 都为BaseLayer的友元
    template <typename T>
    friend class ITLayer;
    virtual void onUpdateParamet() = 0;
};

// 实现层(非抽像层)不会单独从ILayer继承,还一个继承路径应该从BaseLayer来
class ILayer {
   public:
    virtual ~ILayer(){};
    // 请看上面宏AOCE_LAYER_QUERYINTERFACE提供的默认实现
    virtual IBaseLayer* getLayer() = 0;
};

// 分离导致层不同参数的差异(AOCE_LAYER_QUERYINTERFACE)
template <typename T>
class ITLayer : public ILayer {
   protected:
    T oldParamet = {};
    T paramet = {};

   public:
    ITLayer(){};
    virtual ~ITLayer(){};

   public:
    void updateParamet(const T& t) {
        oldParamet = this->paramet;
        this->paramet = t;
        getLayer()->onUpdateParamet();
    };

    T getParamet() { return paramet; }
};

class IPipeGraph {
   public:
    virtual ~IPipeGraph(){};
    virtual GpuType getGpuType() = 0;
    virtual void reset() = 0;
    virtual IBaseLayer* getNode(int32_t index) = 0;
    virtual IBaseLayer* addNode(IBaseLayer* layer) = 0;
    virtual IBaseLayer* addNode(ILayer* layer) = 0;
    virtual bool addLine(int32_t from, int32_t to, int32_t formOut = 0,
                         int32_t toIn = 0) = 0;
    virtual bool addLine(IBaseLayer* from, IBaseLayer* to, int32_t formOut = 0,
                         int32_t toIn = 0) = 0;

    virtual bool getLayerOutFormat(int32_t nodeIndex, int32_t outputIndex,
                                   ImageFormat& format,
                                   bool bOutput = false) = 0;
    virtual bool getLayerInFormat(int32_t nodeIndex, int32_t inputIndex,
                                  ImageFormat& format) = 0;

    // 清除连线(当逻辑变更导致执行列表重组)
    virtual void clearLines() = 0;
    // 清除节点(需要重新变更整个逻辑)
    virtual void clear() = 0;
    virtual bool run() = 0;
};

class IOutputLayerObserver {
   public:
    virtual ~IOutputLayerObserver(){};
    virtual void onImageProcess(uint8_t* data, const ImageFormat& imageFormat,
                                int32_t outIndex) = 0;
};

class IInputLayer : public ITLayer<InputParamet> {
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

class IOutputLayer : public ITLayer<OutputParamet> {
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

// YUV 2 RGBA 转换
typedef ITLayer<YUVParamet> IYUV2RGBALayer;
// RGBA 2 YUV 转换
typedef ITLayer<YUVParamet> IRGBA2YUVLayer;
typedef ITLayer<TexOperateParamet> ITexOperateLayer;
typedef ITLayer<TransposeParamet> ITransposeLayer;
typedef ITLayer<ReSizeParamet> IReSizeLayer;
typedef ITLayer<BlendParamet> IBlendLayer;

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

// 在AoceManager注册vulkan/dx11/cuda类型的LayerFactory
class PipeGraphFactory {
   public:
    virtual ~PipeGraphFactory(){};

   public:
    virtual IPipeGraph* createGraph() = 0;
};

}  // namespace aoce