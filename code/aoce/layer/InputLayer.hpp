#pragma once
#include "BaseLayer.hpp"

namespace aoce {

// inputlayer 应该从VkBaselayer/Dx11Baselayer/CudaBaseLayer继承
class ACOE_EXPORT InputLayer : public IInputLayer {
   public:
    virtual ~InputLayer(){};

   protected:
    // VideoFormat videoFormat = {};
    VideoType videoType = VideoType::rgba8;
    ImageFormat imageFormat = {};
    uint8_t* frameData = nullptr;
    // 用于测试时复制frameData查找数据
    std::vector<uint8_t> videoFrameData;
    int32_t dataSize = -1;

   protected:
    virtual void onDataReady() = 0;
    void dataReady(uint8_t* data, bool bCopy);    
    void checkImageFormat(const ImageFormat& imageFormat);
    void checkImageFormat(const ImageFormat& imageFormat,
                          const VideoType& videoType);

   public:
    // inputCpuData(uint8_t* data)这个版本没有提供长宽,需要这个方法指定
    virtual void setImage(const ImageFormat& newFormat) final;
    virtual void setImage(const VideoFormat& newFormat) final;
    // 输入CPU数据,这个data需要与pipegraph同线程,因为从各方面考虑这个不会复制data里的数据.
    virtual void inputCpuData(uint8_t* data, bool bSeparateRun = false) final;
    virtual void inputCpuData(const VideoFrame& videoFrame,
                              bool bSeparateRun = false) final;
    virtual void inputCpuData(uint8_t* data, const ImageFormat& imageFormat,
                              bool bSeparateRun = false) final;
    virtual void inputGpuData(void* device, void* tex) override;
};

}  // namespace aoce