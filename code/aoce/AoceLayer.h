#pragma once
#include "Aoce.h"

// 导出给外部用户使用,纯净的抽像C++类.
// 请用户不要继承这里的类实现逻辑,找到对应基类实现继承

namespace aoce {

class IBaseLayer;
class IPipeGraph;
class ILayer;

class IBaseLayer {
   public:
    virtual ~IBaseLayer(){};
    virtual const char *getMark() = 0;
    virtual bool bAttachGraph() = 0;
    virtual void setVisable(bool bvisable) = 0;
    virtual void setEnable(bool benable) = 0;
    virtual int32_t getGraphIndex() = 0;
    // 如果层有多个输入,可能不同输入对应不同层内不同层
    // index表示输入节点索引,node表示层内层节点,toInIndex表示对应层内层输入位置
    virtual void setStartNode(IBaseLayer *node, int32_t index = 0,
                              int32_t toInIndex = 0) = 0;
    virtual void setEndNode(IBaseLayer *node) = 0;
    virtual IBaseLayer *addNode(IBaseLayer *layer) = 0;
    virtual IBaseLayer *addNode(ILayer *layer) = 0;
    virtual IBaseLayer *addLine(IBaseLayer *to, int32_t formOut = 0,
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
    virtual IBaseLayer *getLayer() = 0;
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
    void updateParamet(const T &t) {
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
    virtual IBaseLayer *getNode(int32_t index) = 0;
    virtual IBaseLayer *addNode(IBaseLayer *layer) = 0;
    virtual IBaseLayer *addNode(ILayer *layer) = 0;
    virtual bool addLine(int32_t from, int32_t to, int32_t formOut = 0,
                         int32_t toIn = 0) = 0;
    virtual bool addLine(IBaseLayer *from, IBaseLayer *to, int32_t formOut = 0,
                         int32_t toIn = 0) = 0;

    virtual bool getLayerOutFormat(int32_t nodeIndex, int32_t outputIndex,
                                   ImageFormat &format,
                                   bool bOutput = false) = 0;
    virtual bool getLayerInFormat(int32_t nodeIndex, int32_t inputIndex,
                                  ImageFormat &format) = 0;

    // 清除连线(当逻辑变更导致执行列表重组)
    virtual void clearLines() = 0;
    // 清除节点(需要重新变更整个逻辑)
    virtual void clear() = 0;
    virtual bool run() = 0;
};

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
    void *commandbuffer = nullptr;
#if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__) ) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
     void *image = nullptr;
#else
    uint64_t image = 0;
#endif
    int32_t width = 1920;
    int32_t height = 1080;
};

struct GLOutGpuTex{
    int32_t image = -1;
    int32_t width = 1280;
    int32_t height = 720;
};

class IOutputLayerObserver {
   public:
    virtual ~IOutputLayerObserver(){};
    virtual void onImageProcess(uint8_t *data, const ImageFormat &imageFormat,
                                int32_t outIndex) = 0;
    virtual void onFormatChanged(const ImageFormat &imageFormat,
                                 int32_t outIndex){};
};

// 在AoceManager注册vulkan/dx11/cuda类型的LayerFactory
class PipeGraphFactory {
   public:
    virtual ~PipeGraphFactory(){};

   public:
    virtual IPipeGraph *createGraph() = 0;
};

}  // namespace aoce