#pragma once

#include "aoce/AoceCore.h"
#include "aoce_talkto/Talkto.h"
#include <memory>

namespace aoce {
    
using namespace talkto;

#define ENUM_FLAG_OPERATORS(T)                                                 \
    inline T operator~(T a) {                                                  \
        return static_cast<T>(~static_cast<std::underlying_type<T>::type>(a)); \
    }                                                                          \
    inline T operator|(T a, T b) {                                             \
        return static_cast<T>(static_cast<std::underlying_type<T>::type>(a) |  \
                              static_cast<std::underlying_type<T>::type>(b));  \
    }                                                                          \
    inline T operator&(T a, T b) {                                             \
        return static_cast<T>(static_cast<std::underlying_type<T>::type>(a) &  \
                              static_cast<std::underlying_type<T>::type>(b));  \
    }                                                                          \
    inline T operator^(T a, T b) {                                             \
        return static_cast<T>(static_cast<std::underlying_type<T>::type>(a) ^  \
                              static_cast<std::underlying_type<T>::type>(b));  \
    }

enum class ProcessType {
    none = 0,
    source = 1,
    matting = 2,
    transport = 4,
};
ENUM_FLAG_OPERATORS(ProcessType)

typedef ITLayer<TexOperateParamet> ITexOperateLayer;

class VideoProcess : public IVideoDeviceObserver {
	friend class CameraProcess;
   private:
    /* data */
    IPipeGraph* graph = nullptr;
    IVideoDevice* video = nullptr;
    GpuType gpuType = GpuType::other;
    ProcessType processType = ProcessType::none;

    std::unique_ptr<IInputLayer> inputLayer;
    std::unique_ptr<IInputLayer> inputLayer1;
    std::unique_ptr<IOutputLayer> outputLayer;
    std::unique_ptr<IOutputLayer> outputLayer1;
    std::unique_ptr<IOutputLayer> outputLayer2;
    std::unique_ptr<IYUVLayer> yuv2rgbLayer;
	std::unique_ptr<ITexOperateLayer> operateLayer;
    std::unique_ptr<IYUVLayer> rgb2yuvLayer;
    std::unique_ptr<IMattingLayer> mattingLayer;

    VideoFormat format = {};

   public:
    VideoProcess(GpuType gpuType = GpuType::other);
    ~VideoProcess();

   public:
    inline IOutputLayer* getTransportOut() { return outputLayer.get(); }
    inline IOutputLayer* getSourceOut() { return outputLayer1.get(); }
    inline IOutputLayer* getMattingOut() { return outputLayer2.get(); }
    // 请注意在initDevice后调用,会根据device类型来生成不同MattingLayer
    inline IMattingLayer* getMattingLayer() { return mattingLayer.get(); }

   public:
    void initDevice(IVideoDevice* videoPtr, int32_t formatIndex);
    bool openDevice(ProcessType processType);
    void closeDevice();
    bool bOpen();
    VideoFormat getVideoFormat();

   private:
    virtual void onVideoFrame(VideoFrame frame) final;
    virtual void onDepthVideoFrame(VideoFrame frame, VideoFrame depth,
                                   void* alignParamet) final;
};

}  // namespace aoce